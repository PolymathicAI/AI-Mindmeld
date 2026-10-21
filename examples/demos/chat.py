"""Send an editable object-conditioned request to an AI Mindmeld chat server.

Uses only the Python standard library. ``--show-request`` is entirely offline.
The server's model and connector artifacts are required for live generation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def events(lines: Iterable[bytes]) -> Iterator[dict[str, Any]]:
    """Decode JSON data events, joining multiline SSE data and ignoring comments.

    Raises:
        ValueError: for a malformed event or a stream ending inside an event.
    """
    data: list[str] = []
    for raw in lines:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                event = json.loads("\n".join(data))
                if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                    raise ValueError("server event must be an object with a string type")
                yield event
                data = []
        elif line.startswith("data:"):
            value = line[5:]
            data.append(value[1:] if value.startswith(" ") else value)
    if data:
        raise ValueError("server stream ended inside an event")


def run_request(base_url: str, body: dict[str, Any], token: str, timeout: float, transcript: dict[str, Any]) -> None:
    """Stream one answer and retain its events, rejecting errors and incomplete replies.

    Args:
        base_url: server URL without a trailing slash.
        body: request in the chat server's JSON format.
        token: bearer password, or empty for an unauthenticated local deployment.
        timeout: socket timeout in seconds.
        transcript: receives the server configuration, events, and completion status.

    Raises:
        ValueError: for an error event, interrupted stream, or unsuccessful finish.
        URLError: when the server cannot serve the request.
    """
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if token:
        headers["Authorization"] = "Bearer " + token
    config_request = Request(base_url + "/api/config", headers=headers)
    with urlopen(config_request, timeout=timeout) as response:
        transcript["server_config"] = json.load(response)
    request = Request(base_url + "/api/chat", data=json.dumps(body).encode("utf-8"), headers=headers)
    with urlopen(request, timeout=timeout) as response:
        if response.headers.get_content_type() != "text/event-stream":
            raise ValueError("expected a text/event-stream response from /api/chat")
        for event in events(response):
            transcript["events"].append(event)
            kind = event["type"]
            if kind == "content":
                sys.stdout.write(event["delta"])
                sys.stdout.flush()
            elif kind == "status" and event.get("detail"):
                sys.stderr.write(str(event["detail"]) + "\n")
            elif kind == "error":
                raise ValueError("server error: " + str(event.get("message", "unknown error")))
            elif kind == "done":
                reason = event.get("finish_reason")
                if reason not in {"stop", "length"}:
                    raise ValueError(f"generation did not complete: {reason!r}")
                transcript["complete"] = True
                sys.stdout.write("\n")
                if reason == "length":
                    sys.stderr.write("Answer reached the requested token limit.\n")
                return
    raise ValueError("server stream ended without a done event")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments and reject incompatible modes or invalid timeouts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path, nargs="?", help="JSON request file")
    parser.add_argument("--url", default="http://127.0.0.1:8811", help="chat server URL")
    parser.add_argument("--password-file", type=Path, help="file containing the server password")
    parser.add_argument("--timeout", type=float, default=900, help="socket timeout in seconds")
    parser.add_argument("--output", type=Path, help="save a transcript to a new file")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="show server and connector configuration")
    mode.add_argument("--show-request", action="store_true", help="print the request without contacting a server")
    args = parser.parse_args(argv)
    if args.list and (args.request or args.output):
        parser.error("--list does not take a request or --output")
    if not args.list and args.request is None:
        parser.error("provide a request file, or use --list")
    if args.show_request and args.output:
        parser.error("--output saves live transcripts; use shell redirection for --show-request")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    """Preview a request, inspect a deployment, or stream a live answer."""
    args = parse_args(argv)
    output = None
    transcript: dict[str, Any] = {"complete": False, "events": []}
    try:
        body = None if args.list else json.loads(args.request.read_text(encoding="utf-8"))
        if not args.list and not isinstance(body, dict):
            raise ValueError("request must be a JSON object")
        if args.show_request:
            sys.stdout.write(json.dumps(body, indent=2, ensure_ascii=False) + "\n")
            return 0
        parsed = urlsplit(args.url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("--url must be an HTTP or HTTPS server URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("--url must not contain credentials, a query, or a fragment")
        base_url = args.url.rstrip("/")
        token = args.password_file.read_text(encoding="utf-8").strip() if args.password_file else ""
        if args.password_file and not token:
            raise ValueError("password file is empty")
        if args.list:
            headers = {"Authorization": "Bearer " + token} if token else {}
            with urlopen(Request(base_url + "/api/config", headers=headers), timeout=args.timeout) as response:
                sys.stdout.write(json.dumps(json.load(response), indent=2) + "\n")
            return 0
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            output = args.output.open("x", encoding="utf-8")
        transcript.update({"request": body, "server": base_url, "started_at": datetime.now(UTC).isoformat()})
        run_request(base_url, body, token, args.timeout, transcript)
        return 0
    except (OSError, URLError, ValueError) as error:
        transcript["error"] = str(error)
        sys.stderr.write(f"Demo failed: {error}\n")
        return 1
    except KeyboardInterrupt:
        transcript["error"] = "interrupted by user"
        sys.stderr.write("\nDemo interrupted.\n")
        return 130
    finally:
        if output:
            with output:
                json.dump(transcript, output, indent=2, ensure_ascii=False)
                output.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
