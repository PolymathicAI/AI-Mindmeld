"""A deployment's smoke test over HTTP, and its receipt.

``python -m latent_interconnect.chat.smoke --config <yaml> --url http://<host>:<port> --out <receipt.json>
(--password-file F | --no-auth) [--revision <commit>] [-- <server command>]``. The client of the API contract
(``examples/guides/chat.md`` §4), on the standard library and the package's config only. Given a server command (the
smoke job's ``launch serve``), it starts it in its own process group, stops it at the end (SIGTERM, then SIGKILL after
:data:`STOP_GRACE_S`) and records its exit status; a server that exits before it is ready ends the wait at once
("server exited (rc=N)"). It waits until ``/api/health`` is ready (or start-up failed, the server exited, or
``ready_timeout_s`` passed), then:

1. **Access** (with a password; ``--no-auth`` skips it): ``/api/config`` without credentials is 401, ``/api/login``
   with the password sets the session cookie, and the cookie alone opens ``/api/config``.
2. **Cached ids:** for every connector the deployment lists, one request whose placeholder is the domain's object id
   (:attr:`SmokeInput.id`); its injection must come from the run's cached embeddings (``source: cache``).
3. **Raw values:** for every connector, one request whose placeholder is the domain's raw value, encoded live
   (``source: live``), in :attr:`SmokeConfig.order`.

Every request carries ``Authorization: Bearer <password>`` and runs alone (concurrency 1). A request passes when it is
HTTP 200, its stream ends with ``done`` (not ``error``), ``finish_reason`` is ``stop`` or ``length`` with at least one
completion token, and every injection has the expected source. A failure does not stop the run: one attempt gathers
all the evidence it can. The receipt (JSON) is written before the wait and rewritten after every step, so a run
killed midway keeps what it did; it records ``--revision`` (the code the job runs). Exit status: 0 when every step
passed, 1 when any failed, :data:`NOT_READY` (3) when the deployment never became ready; 2 is a usage error.
"""

from __future__ import annotations

import argparse
import contextlib
import http.client
import json
import os
import signal
import subprocess
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import Field

from latent_interconnect.config import Config, load_config

__all__ = ["SmokeConfig", "SmokeInput", "SmokeSampling", "main", "run_smoke"]

NOT_READY = 3
"""The exit status when the deployment never became ready (2 is argparse's usage error)."""
HEALTH_TIMEOUT_S = 10.0
"""The socket timeout of one ``/api/health`` poll."""
STOP_GRACE_S = 60.0
"""How long a server the client started has to stop after SIGTERM before its process group is killed."""


class SmokeInput(Config):
    """The objects one domain's requests use.

    Attributes:
        id: an object id (the chat config's ``inputs.<domain>.id`` type) that every run of the domain holds in its
            cached embeddings, so its request is served from the cache.
        value: a raw value (the ``inputs.<domain>.value`` type), always encoded live.
    """

    id: str = Field(min_length=1)
    value: str = Field(min_length=1)


class SmokeSampling(Config):
    """The sampling of every smoke request (the server's ``sampling`` body).

    Attributes:
        temperature: sampling temperature (0 = greedy).
        max_tokens: the answer's token cap.
        seed: the sampling seed.
    """

    temperature: float = Field(ge=0.0, le=2.0)
    max_tokens: int = Field(ge=1)
    seed: int = 0


class SmokeConfig(Config):
    """A smoke test of one chat deployment.

    Attributes:
        inputs: domain name → the objects its requests use (every domain the deployment's connectors serve).
        prompts: domain name → the user turn, holding the placeholder ``[[P1]]``.
        sampling: every request's sampling.
        thinking: thinking mode.
        order: connector ids whose raw-value requests go first, in this order; the deployment's other connectors
            follow in ``/api/config``'s order. Ids the deployment does not list are ignored, so one order serves
            every receiver.
        ready_timeout_s: how long to wait for ``/api/health`` to report ready.
        request_timeout_s: the longest one request may take (socket timeout, including a cold encoder's start).
    """

    inputs: dict[str, SmokeInput] = Field(min_length=1)
    prompts: dict[str, str] = Field(min_length=1)
    sampling: SmokeSampling
    thinking: bool = False
    order: tuple[str, ...] = ()
    ready_timeout_s: float = Field(gt=0)
    request_timeout_s: float = Field(gt=0)


class _Http:
    """One base URL, its password and timeouts; each call opens its own connection."""

    def __init__(self, url: str, password: str | None, timeout_s: float) -> None:
        parts = urlsplit(url)
        if parts.scheme != "http" or not parts.hostname:
            raise ValueError(f"--url {url!r}: expected http://<host>:<port>")
        self.host, self.port = parts.hostname, parts.port or 80
        self.password, self.timeout_s = password, timeout_s

    def call(
        self, method: str, path: str, body: Any = None, *, bearer: bool = True, cookie: str | None = None
    ) -> tuple[int, dict[str, str], http.client.HTTPConnection, http.client.HTTPResponse]:
        headers = {"Content-Type": "application/json"}
        if bearer and self.password is not None:
            headers["Authorization"] = f"Bearer {self.password}"
        if cookie is not None:
            headers["Cookie"] = cookie
        conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout_s)
        conn.request(method, path, body=None if body is None else json.dumps(body), headers=headers)
        response = conn.getresponse()
        return response.status, {k.lower(): v for k, v in response.getheaders()}, conn, response

    def json(self, method: str, path: str, body: Any = None, **kw: Any) -> tuple[int, dict[str, str], Any]:
        status, headers, conn, response = self.call(method, path, body, **kw)
        text = response.read().decode(errors="replace")
        conn.close()
        try:
            return status, headers, json.loads(text)
        except json.JSONDecodeError:
            return status, headers, text


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _wait_ready(
    api: _Http, timeout_s: float, server: subprocess.Popen[bytes] | None = None
) -> tuple[dict[str, Any] | None, float]:
    """The last ``/api/health`` answer once ready or failed (None if it never answered), and the seconds waited.

    The wait also ends when ``server`` (the server process the client started) has exited.
    """
    t0, health = time.monotonic(), None
    while time.monotonic() - t0 < timeout_s:
        if server is not None and server.poll() is not None:
            break
        try:
            _, _, body = api.json("GET", "/api/health", bearer=False)
            health = body if isinstance(body, dict) else None
        except (OSError, http.client.HTTPException):
            health = None
        if health and (health.get("ready") or health.get("startup", {}).get("failed")):
            break
        time.sleep(2.0)
    return health, round(time.monotonic() - t0, 1)


def _access(api: _Http) -> dict[str, Any]:
    """The password and cookie paths: refused without credentials, the cookie from ``/api/login`` accepted."""
    bare, _, _ = api.json("GET", "/api/config", bearer=False)
    login, headers, _ = api.json("POST", "/api/login", {"token": api.password}, bearer=False)
    cookie = headers.get("set-cookie", "").split(";", 1)[0]
    with_cookie, _, _ = api.json("GET", "/api/config", bearer=False, cookie=cookie) if cookie else (None, {}, None)
    result = {"no_credentials": bare, "login": login, "cookie_set": bool(cookie), "with_cookie": with_cookie}
    passed = (
        bare == HTTPStatus.UNAUTHORIZED and login == HTTPStatus.OK and bool(cookie) and with_cookie == HTTPStatus.OK
    )
    return result | {"passed": passed, "why": None if passed else "expected 401, 200 with a cookie, then 200"}


def _request(api: _Http, cfg: SmokeConfig, connector: dict[str, Any], path: str) -> dict[str, Any]:
    """One chat request for ``connector`` by its domain's cached id (``path == "cached"``) or raw value."""
    domain = connector["domain"]
    if domain not in cfg.inputs or domain not in cfg.prompts:
        return {"connector": connector["id"], "path": path, "passed": False, "why": f"no smoke input for {domain!r}"}
    id_type, value_type = connector["input_types"]
    obj = cfg.inputs[domain]
    input_type, value, expected = (id_type, obj.id, "cache") if path == "cached" else (value_type, obj.value, "live")
    body = {
        "messages": [{"role": "user", "content": cfg.prompts[domain]}],
        "placeholders": {"P1": {"connector": connector["id"], "input_type": input_type, "value": value}},
        "thinking": cfg.thinking,
        "sampling": cfg.sampling.model_dump(),
    }
    out: dict[str, Any] = {"connector": connector["id"], "path": path, "input_type": input_type, "value": value}
    out |= {"expected_source": expected, "sent": _now()}
    t0, first, text, events = time.monotonic(), None, [], []
    try:
        status, _, conn, response = api.call("POST", "/api/chat", body)
        out["http_status"] = status
        if status != HTTPStatus.OK:
            out["body"] = response.read().decode(errors="replace")[:2000]
        else:
            for raw in response:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                event = json.loads(line[5:])
                if event.get("type") == "reasoning":
                    continue
                if event.get("type") == "content":
                    if event.get("delta") and first is None:
                        first = time.monotonic() - t0
                    text.append(event.get("delta", ""))
                    continue
                events.append({"t": round(time.monotonic() - t0, 3), **event})
                if event.get("type") in ("done", "error"):
                    break
        conn.close()
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as err:
        out["exception"] = repr(err)
    out |= {"first_visible_client_s": None if first is None else round(first, 3)}
    out |= {"total_client_s": round(time.monotonic() - t0, 3), "events": events, "text": "".join(text)}
    out["why"] = _verdict(out, events, expected)
    out["passed"] = out["why"] is None
    return out


def _verdict(out: dict[str, Any], events: list[dict[str, Any]], expected: str) -> str | None:
    """None if the request passed, else why it did not."""
    if out.get("http_status") != HTTPStatus.OK:
        return f"HTTP {out.get('http_status')}" + (f" ({out['exception']})" if "exception" in out else "")
    last = events[-1] if events else {}
    if last.get("type") != "done":
        return f"no done event: {last.get('message') or out.get('exception') or 'the stream ended'}"
    if last.get("finish_reason") not in ("stop", "length") or not last.get("usage", {}).get("completion_tokens"):
        return f"finish_reason {last.get('finish_reason')!r} with {last.get('usage')}"
    sources = {k: v.get("source") for k, v in (last.get("injections") or {}).items()}
    if not sources or set(sources.values()) != {expected}:
        return f"injection sources {sources}, expected {expected!r}"
    return None


def _raw_order(connectors: list[dict[str, Any]], order: Sequence[str]) -> list[dict[str, Any]]:
    rank = {cid: i for i, cid in enumerate(order)}
    return sorted(connectors, key=lambda c: (rank.get(c["id"], len(rank)), connectors.index(c)))


def _group_alive(server: subprocess.Popen[bytes]) -> bool:
    if server.poll() is None:
        return True
    try:
        os.killpg(server.pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _stop(server: subprocess.Popen[bytes]) -> int:
    """Stop the server's process group: SIGTERM, then SIGKILL after :data:`STOP_GRACE_S`; return its exit status."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(server.pid, signal.SIGTERM)
    deadline = time.monotonic() + STOP_GRACE_S
    while _group_alive(server) and time.monotonic() < deadline:
        time.sleep(0.2)
    if _group_alive(server):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(server.pid, signal.SIGKILL)
    return server.wait()


def run_smoke(
    cfg: SmokeConfig,
    url: str,
    password_file: Path | None,
    out: Path,
    *,
    server: Sequence[str] = (),
    revision: str | None = None,
) -> int:
    """Run the smoke test against ``url`` and write its receipt to ``out``; return the exit status.

    Args:
        cfg: the smoke config.
        url: the deployment's base URL (``http://<host>:<port>``).
        password_file: the shared password's file, read once the deployment is ready (the server makes it if it is
            absent; the client never does). None: a ``--no-auth`` deployment, whose access checks are skipped.
        out: the receipt file (rewritten after every step).
        server: the command of the server to start, test and stop (empty: one already running at ``url``).
        revision: the code revision the deployment runs, recorded in the receipt.
    """
    receipt: dict[str, Any] = {"url": url, "revision": revision, "started": _now()}
    receipt |= {"config": cfg.model_dump(mode="json"), "steps": []}

    def save() -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(receipt, indent=1) + "\n")

    save()
    process = subprocess.Popen(list(server), start_new_session=True) if server else None
    try:
        return _run_smoke(cfg, url, password_file, receipt=receipt, save=save, server=process)
    finally:
        if process is not None:
            receipt["server"] = {"command": list(server), "exit": _stop(process)}
            save()


def _run_smoke(
    cfg: SmokeConfig,
    url: str,
    password_file: Path | None,
    *,
    receipt: dict[str, Any],
    save: Callable[[], None],
    server: subprocess.Popen[bytes] | None,
) -> int:
    api = _Http(url, None, cfg.request_timeout_s)
    health, waited = _wait_ready(
        _Http(url, None, min(HEALTH_TIMEOUT_S, cfg.request_timeout_s)), cfg.ready_timeout_s, server
    )
    receipt |= {"ready_wait_s": waited, "health_at_ready": health}
    if not (health and health.get("ready")):
        why = "the deployment did not become ready"
        if server is not None and server.poll() is not None:
            why = f"server exited (rc={server.returncode}) before it became ready"
        receipt |= {"passed": False, "why": why, "finished": _now()}
        save()
        return NOT_READY
    if password_file is not None:
        try:
            api.password = password_file.expanduser().read_text().strip()
        except OSError as err:
            receipt |= {"passed": False, "why": f"the password: {err}", "finished": _now()}
            save()
            return 1
    status, _, view = api.json("GET", "/api/config")
    if status != HTTPStatus.OK or not isinstance(view, dict):
        receipt |= {"passed": False, "why": f"/api/config with the password: HTTP {status}", "finished": _now()}
        save()
        return 1
    receipt["deployment"] = {k: view.get(k) for k in ("model", "receiver", "engine", "max_model_len", "connectors")}
    save()
    if api.password is not None:
        receipt["steps"].append({"step": "access", **_access(api)})
        save()
    connectors = view["connectors"]
    for path, plan in (("cached", connectors), ("raw", _raw_order(connectors, cfg.order))):
        for connector in plan:
            receipt["steps"].append({"step": "chat", **_request(api, cfg, connector, path)})
            save()
    failed = [" ".join(str(s[k]) for k in ("step", "connector", "path") if k in s) for s in receipt["steps"]]
    failed = [name for name, s in zip(failed, receipt["steps"], strict=True) if not s["passed"]]
    receipt |= {"passed": not failed, "failed": failed, "finished": _now()}
    save()
    return 0 if not failed else 1


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m latent_interconnect.chat.smoke``; return the exit status (see the module docstring)."""
    parser = argparse.ArgumentParser(prog="python -m latent_interconnect.chat.smoke")
    parser.add_argument("--config", type=Path, required=True, help="the smoke config (configs/chat/smoke.yaml)")
    parser.add_argument("--url", required=True, help="the deployment, http://<host>:<port>")
    parser.add_argument("--out", type=Path, required=True, help="the receipt (JSON)")
    parser.add_argument("--password-file", type=Path, default=None, help="the shared password's file")
    parser.add_argument("--no-auth", action="store_true", help="a deployment without a password (no access checks)")
    parser.add_argument("--revision", default=None, help="the code revision the deployment runs (for the receipt)")
    parser.add_argument("server", nargs="*", help="after --: the server command to start, test and stop")
    args = parser.parse_args(argv)
    if args.no_auth == (args.password_file is not None):
        parser.error("give one of --password-file and --no-auth")
    cfg = load_config(args.config, SmokeConfig)
    return run_smoke(cfg, args.url, args.password_file, args.out, server=args.server, revision=args.revision)


if __name__ == "__main__":
    raise SystemExit(main())
