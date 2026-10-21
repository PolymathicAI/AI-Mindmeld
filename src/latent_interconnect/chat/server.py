"""The chat HTTP server: FastAPI over :class:`~latent_interconnect.chat.service.ChatService`, and the static UI.

Serves ``chat/static/`` at ``/`` and the API at ``/api/*`` (the contract: ``examples/guides/chat.md``). Every ``/api/*``
route except ``/api/health`` and ``/api/login`` needs ``Authorization: Bearer <password>`` or the ``chat_session``
cookie that ``POST /api/login`` sets (an HMAC of the shared password); ``--no-auth`` is allowed only on a loopback
address, and a wildcard bind is refused. Conversations are not stored: the client sends the whole
conversation each turn. ``/api/chat`` answers with server-sent events (HTTP 200; refusals are ``error`` events); a body
that is not a JSON object is HTTP 400, a missing or wrong password 401, a body over the cap 413.

A request's events are produced in a worker thread (generation blocks) and relayed as they come, each engine step's
text delta as its own event; a comment line keeps the connection alive while nothing is produced. A client that goes
away, or ``POST /api/stop``, sets the request's stop event, so the generation stream is closed and cancelled.
"""

import asyncio
import hashlib
import hmac
import ipaddress
import json
import secrets
import uuid
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from latent_interconnect.chat.service import REQUEST_ID, ChatService

__all__ = ["MAX_BODY_BYTES", "STATIC", "BodyTooLargeError", "body_of", "check_bind", "create_app", "is_loopback"]

STATIC = Path(__file__).resolve().parent / "static"
"""The UI (plain JavaScript, no build step)."""
MAX_BODY_BYTES = 2 * 1024 * 1024
"""The largest request body accepted."""
KEEPALIVE_S = 15.0
"""Seconds without an event before a keep-alive comment is sent."""
SESSION_COOKIE = "chat_session"


def is_loopback(host: str) -> bool:
    """True only for a loopback address (127.0.0.0/8, ::1, ``localhost``)."""
    h = host.strip().strip("[]")
    if h == "localhost":
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def check_bind(host: str, no_auth: bool) -> None:
    """Refuse a wildcard bind, and ``no_auth`` on anything but loopback.

    Raises:
        ValueError: naming the refused address.
    """
    if host.strip().strip("[]") in {"", "0.0.0.0", "::"}:  # refused, never bound
        raise ValueError(f"a wildcard bind ({host!r}) is refused: give the address to serve on")
    if no_auth and not is_loopback(host):
        raise ValueError(f"no-auth mode serves loopback only, not {host!r}; reach it through an SSH tunnel")


def create_app(
    service: ChatService, password: str | None, *, no_auth: bool = False, secure_cookie: bool = False
) -> Any:
    """Return the FastAPI app (FastAPI is imported here: the ``.venv`` and ``vllm`` environments carry it).

    Args:
        service: the chat service.
        password: the shared password (a short one is fine); ignored with ``no_auth``.
        no_auth: serve without a password (the caller checks the bind is loopback, :func:`check_bind`).
        secure_cookie: mark the session cookie ``Secure``.

    Raises:
        ValueError: without a password unless ``no_auth``.
    """
    from fastapi import FastAPI, Request  # noqa: PLC0415
    from fastapi.responses import JSONResponse, StreamingResponse  # noqa: PLC0415
    from fastapi.staticfiles import StaticFiles  # noqa: PLC0415

    if not no_auth and not password:
        raise ValueError("a non-empty shared password is needed (or no-auth mode on a loopback address)")
    auth = _Auth(password or secrets.token_urlsafe(24), no_auth)
    executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="chat")
    app = FastAPI(title="latent_interconnect chat", docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(BodyTooLargeError)
    async def too_large(_request: Request, _err: BodyTooLargeError) -> Any:
        return JSONResponse({"error": f"request body over {MAX_BODY_BYTES} bytes"}, status_code=413)

    @app.middleware("http")
    async def guard(request: Request, call_next: Any) -> Any:
        path = request.url.path
        if path.startswith("/api/") and path not in {"/api/health", "/api/login"} and not auth.ok(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        length = request.headers.get("content-length")
        if length is not None and (not length.isdigit() or int(length) > MAX_BODY_BYTES):
            return JSONResponse({"error": f"request body over {MAX_BODY_BYTES} bytes"}, status_code=413)
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response

    def bad() -> Any:
        return JSONResponse({"error": "the body must be a JSON object"}, status_code=400)

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return service.health()

    @app.post("/api/login")
    async def login(request: Request) -> Any:
        given = (await body_of(request) or {}).get("token", "")
        if not no_auth and not (isinstance(given, str) and _same(given, auth.token)):
            return JSONResponse({"ok": False, "error": "invalid password"}, status_code=401)
        response = JSONResponse({"ok": True})
        response.set_cookie(
            SESSION_COOKIE,
            auth.session.decode(),
            httponly=True,
            samesite="strict",
            secure=secure_cookie,
            max_age=7 * 24 * 3600,
            path="/",
        )
        return response

    @app.post("/api/logout")
    async def logout() -> Any:
        response = JSONResponse({"ok": True})
        response.delete_cookie(SESSION_COOKIE, path="/")
        return response

    @app.get("/api/config")
    async def config() -> dict[str, Any]:
        return service.config_view()

    @app.post("/api/placeholder/check")
    async def check(request: Request) -> Any:
        body = await body_of(request)
        if body is None:
            return bad()
        return await asyncio.get_running_loop().run_in_executor(executor, service.check, body)

    @app.post("/api/chat")
    async def chat(request: Request) -> Any:
        body = await body_of(request)
        if body is None:
            return bad()
        rid = "chat-" + uuid.uuid4().hex[:12]
        return StreamingResponse(
            _relay(service, body, rid, executor),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/stop")
    async def stop(request: Request) -> Any:
        rid = (await body_of(request) or {}).get("request_id")
        if not isinstance(rid, str) or not REQUEST_ID.fullmatch(rid):
            return JSONResponse({"ok": False, "error": "request_id must match " + REQUEST_ID.pattern}, status_code=400)
        service.stop(rid)
        return {"ok": True}

    app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="static")
    return app


_END = object()


def _same(given: str, expected: bytes) -> bool:
    """Constant-time comparison (on bytes: a non-ASCII string would make ``compare_digest`` raise)."""
    return hmac.compare_digest(given.encode("utf-8", "surrogatepass"), expected)


class _Auth:
    """The shared password and its session cookie (an HMAC of it), as the original app has them."""

    def __init__(self, password: str, no_auth: bool) -> None:
        self.token, self.no_auth = password.encode(), no_auth
        self.session = hmac.new(self.token, SESSION_COOKIE.encode(), hashlib.sha256).hexdigest().encode()

    def ok(self, request: Any) -> bool:
        """Whether a request carries the password (bearer) or the session cookie."""
        if self.no_auth:
            return True
        header = request.headers.get("authorization", "")
        if header.lower().startswith("bearer ") and _same(header[7:].strip(), self.token):
            return True
        cookie = request.cookies.get(SESSION_COOKIE, "")
        return bool(cookie) and _same(cookie, self.session)


class BodyTooLargeError(ValueError):
    """A request body over :data:`MAX_BODY_BYTES` (HTTP 413)."""


async def body_of(request: Any) -> dict[str, Any] | None:
    """The request's JSON object body, or None (not JSON, not an object, NaN or Infinity).

    Raises:
        BodyTooLargeError: past :data:`MAX_BODY_BYTES` (a chunked body has no length the middleware could check).
    """
    raw = b""
    async for chunk in request.stream():
        raw += chunk
        if len(raw) > MAX_BODY_BYTES:
            raise BodyTooLargeError
    try:
        body = json.loads(raw, parse_constant=_no_constant)
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


async def _relay(
    service: ChatService, body: dict[str, Any], rid: str, executor: ThreadPoolExecutor
) -> AsyncIterator[str]:
    """Run the request's events in a worker thread and pass each on as it comes; stop the request if the client goes."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[object] = asyncio.Queue()
    stop = service.stop_event(rid)

    def work() -> None:
        events = service.events(body, rid)
        try:
            for event in events:
                loop.call_soon_threadsafe(queue.put_nowait, event)
        finally:
            events.close()
            loop.call_soon_threadsafe(queue.put_nowait, _END)

    loop.run_in_executor(executor, work)
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), KEEPALIVE_S)
            except TimeoutError:
                yield ": keepalive\n\n"
                continue
            if event is _END:
                return
            yield f"data: {json.dumps(event)}\n\n"
    finally:
        stop.set()  # a client that went away stops its request; a finished one is unaffected


def _no_constant(name: str) -> None:
    raise ValueError(f"{name} is not JSON")
