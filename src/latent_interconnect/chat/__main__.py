"""``python -m latent_interconnect.chat --config <yaml> --receiver <id> --host <addr> --port <n> [...]``.

Serves one receiver of a chat config (``chat.config``) until stopped. The workspace is ``LIP_DATA``'s
(``Workspace.from_env``): pins and the production store, read only; everything chat writes (staged inputs, live
embeddings, write banks) goes to ``--store`` (default ``$LIP_DATA/chat/store``), a disposable store. The receiver
loads in the background while the server already answers (``/api/health`` shows its progress). The password is
``--password-file``'s (created with a random one, mode 600, if absent), else ``CHAT_TOKEN``; ``--no-auth`` serves
without one on a loopback address only. ``--python ENV=PATH`` gives an encoder environment's interpreter (the site
launcher passes the environment table's); ``--option K=V`` an execution option of the generation backend.
``--max-tokens-cap`` and ``--max-prompt-tokens`` narrow what is served to a measured envelope and ``--serial-requests``
serves one whole request at a time, live encodings included (the site launcher passes all three for a local
deployment); without them the config's cap and the receiver's context are the only limits, and only generation is one
request at a time (another request's live encoding can overlap it). ``--max-resident-encoders N`` keeps at most N
live encoder workers up, one per recipe, stopping the least recently used idle one first (``chat.encoding``; the site
launcher passes 1 for a local deployment); without it every encoder used stays resident. ``--yes-no ANSWER.yaml``
gives the receiver's answer convention (``configs/answers/``), which offers Yes/No requests where the receiver's score
backend is its generate backend (``chat.service``). Ctrl-C or SIGTERM stops the server: it shuts down, stops every
encoder worker and exits 0 (1 if the start-up failed).
"""

from __future__ import annotations

import argparse
import os
import secrets
import signal
import sys
import threading
from collections.abc import Sequence
from pathlib import Path

from latent_interconnect.artifacts.store import ArtifactStore
from latent_interconnect.chat.config import ChatConfig
from latent_interconnect.chat.encoding import EncoderPool
from latent_interconnect.chat.server import check_bind, create_app
from latent_interconnect.chat.service import ChatService
from latent_interconnect.config import load_config
from latent_interconnect.data.answers import AnswerConvention
from latent_interconnect.workspace import Workspace

__all__ = ["main", "read_password"]


def _pairs(items: Sequence[str], what: str) -> dict[str, str]:
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not (sep and key and value):
            raise SystemExit(f"{what} {item!r}: expected KEY=VALUE")
        out[key] = value
    return out


def read_password(path: Path | None) -> str:
    """Return the shared password: the file's (made with a random one, mode 600, if absent), else ``CHAT_TOKEN``."""
    if path is not None:
        path = path.expanduser()
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as fh:
                fh.write(secrets.token_urlsafe(24) + "\n")
            sys.stderr.write(f"[chat] wrote a new password to {path} (mode 600)\n")
        return path.read_text().strip()
    password = os.environ.get("CHAT_TOKEN", "")
    if not password:
        raise SystemExit("give --password-file or CHAT_TOKEN (or --no-auth on a loopback address)")
    return password


def main(argv: Sequence[str] | None = None) -> int:
    """Run the chat server; return the process exit code."""
    parser = argparse.ArgumentParser(prog="python -m latent_interconnect.chat")
    parser.add_argument("--config", type=Path, required=True, help="the chat config (configs/chat/<site>.yaml)")
    parser.add_argument("--receiver", required=True, help="the receiver id to serve")
    parser.add_argument("--host", required=True, help="the address to serve on (no wildcard)")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--password-file", type=Path, default=None, help="the shared password's file")
    parser.add_argument("--no-auth", action="store_true", help="no password (loopback address only)")
    parser.add_argument("--secure-cookie", action="store_true", help="mark the session cookie Secure")
    parser.add_argument("--python", action="append", default=[], help="ENV=PATH: an encoder environment's interpreter")
    parser.add_argument("--encoder-device", default="cuda", help="the torch device the live encoders run on")
    parser.add_argument("--option", action="append", default=[], help="K=V: a generation backend execution option")
    parser.add_argument("--deployment", default=None, help="the id /api/health reports (the front door's lease)")
    parser.add_argument("--store", type=Path, default=None, help="chat's own store (default $LIP_DATA/chat/store)")
    parser.add_argument("--max-tokens-cap", type=int, default=None, help="a lower answer cap than the config's")
    parser.add_argument("--max-prompt-tokens", type=int, default=None, help="the longest rendered prompt served")
    parser.add_argument(
        "--serial-requests", action="store_true", help="one whole request at a time (encoding and generation)"
    )
    parser.add_argument(
        "--max-resident-encoders", type=int, default=None, help="at most N live encoder workers (one per recipe)"
    )
    parser.add_argument(
        "--yes-no", type=Path, default=None, metavar="ANSWER.yaml", help="the receiver's answer convention (Yes/No)"
    )
    args = parser.parse_args(argv)
    if args.max_resident_encoders is not None and args.max_resident_encoders < 1:
        parser.error("--max-resident-encoders must be at least 1 (omit it for no limit)")
    try:
        check_bind(args.host, args.no_auth)
    except ValueError as err:
        parser.error(str(err))
    password = None if args.no_auth else read_password(args.password_file)
    answer = None if args.yes_no is None else load_config(args.yes_no, AnswerConvention)
    ws = Workspace.from_env()
    config = ws.load(args.config, ChatConfig)
    store = ArtifactStore(args.store or ws.paths.data / "chat" / "store")
    pool = EncoderPool(
        store, _pairs(args.python, "--python"), args.encoder_device, max_resident=args.max_resident_encoders
    )
    service = ChatService(
        config,
        args.receiver,
        ws,
        pool,
        deployment=args.deployment,
        options=_pairs(args.option, "--option"),
        max_tokens_cap=args.max_tokens_cap,
        max_prompt_tokens=args.max_prompt_tokens,
        serial_requests=args.serial_requests,
        answer=answer,
    )
    app = create_app(service, password, no_auth=args.no_auth, secure_cookie=args.secure_cookie)
    threading.Thread(target=service.load, daemon=True, name="chat-load").start()
    import uvicorn  # noqa: PLC0415  (the .venv and vllm environments carry it)

    # uvicorn shuts down on Ctrl-C or SIGTERM and then raises the signal again; with this handler the re-raised
    # SIGTERM becomes the KeyboardInterrupt uvicorn.run absorbs, as Ctrl-C's does, instead of ending the process before
    # the encoder workers stop
    previous = signal.signal(signal.SIGTERM, _stop)
    try:
        try:
            uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        finally:
            pool.close()  # every encoder worker, before the process exits
    finally:
        signal.signal(signal.SIGTERM, previous)  # also after a second signal during pool.close
    return 0 if service.failed is None else 1


def _stop(_signum: int, _frame: object) -> None:
    """SIGTERM as Ctrl-C: uvicorn's re-raised SIGTERM would otherwise end the process before the encoders stop."""
    raise KeyboardInterrupt


if __name__ == "__main__":
    raise SystemExit(main())
