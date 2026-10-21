"""Exercise the demo's HTTP boundary without model loading or network access."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from unittest.mock import patch

from examples.demos.chat import events, main


class Response(io.BytesIO):
    """A byte stream with the response headers consumed by the client."""

    def __init__(self, data: bytes, content_type: str) -> None:
        super().__init__(data)
        self.headers = Message()
        self.headers["Content-Type"] = content_type


def stream(*items: dict[str, object]) -> Response:
    """Build an SSE fixture, including the server's keepalive comment."""
    data = b": keepalive\n\n" + b"".join(
        b"data: " + json.dumps(item, ensure_ascii=False).encode() + b"\n\n" for item in items
    )
    return Response(data, "text/event-stream")


class DemoTest(unittest.TestCase):
    """Check completion, errors, offline preview, and transcript preservation."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.request = self.root / "request.json"
        self.request.write_text('{"messages": [], "placeholders": {}}', encoding="utf-8")
        self.output = self.root / "answer.json"
        self.password = self.root / "password"
        self.password.write_text("fixture-password\n", encoding="utf-8")

    def invoke(self, response: Response) -> tuple[int, str, str]:
        """Run a live-mode client against canned config and SSE responses."""
        config = Response(b'{"receiver": "fixture"}', "application/json")
        with patch("examples.demos.chat.urlopen", side_effect=[config, response]) as opened:
            with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
                code = main([str(self.request), "--output", str(self.output), "--password-file", str(self.password)])
            self.assertEqual(opened.call_args_list[0].args[0].full_url, "http://127.0.0.1:8811/api/config")
            sent = opened.call_args_list[1].args[0]
            self.assertEqual(sent.get_method(), "POST")
            self.assertEqual(sent.get_header("Authorization"), "Bearer fixture-password")
            self.assertEqual(json.loads(sent.data), json.loads(self.request.read_text()))
        return code, out.getvalue(), err.getvalue()

    def test_answer_and_reasoning_stay_separate_and_password_is_not_saved(self) -> None:
        code, text, _ = self.invoke(
            stream(
                {"type": "reasoning", "delta": "fixture reasoning"},
                {"type": "content", "delta": "αβ fixture"},
                {"type": "done", "finish_reason": "stop"},
            )
        )
        self.assertEqual(code, 0)
        self.assertEqual(text, "αβ fixture\n")
        saved = json.loads(self.output.read_text())
        self.assertTrue(saved["complete"])
        self.assertEqual(saved["events"][0]["type"], "reasoning")
        self.assertNotIn("fixture-password", self.output.read_text())

    def test_partial_response_is_saved_but_does_not_report_success(self) -> None:
        code, text, error = self.invoke(stream({"type": "content", "delta": "partial"}))
        self.assertEqual(code, 1)
        self.assertEqual(text, "partial")
        self.assertIn("without a done event", error)
        saved = json.loads(self.output.read_text())
        self.assertFalse(saved["complete"])
        self.assertEqual(len(saved["events"]), 1)

    def test_server_error_and_abort_fail(self) -> None:
        for final in (
            {"type": "error", "message": "connector unavailable"},
            {"type": "done", "finish_reason": "abort"},
        ):
            with self.subTest(final=final):
                self.output = self.root / (final["type"] + ".json")
                code, _, _ = self.invoke(stream(final))
                self.assertEqual(code, 1)
                self.assertFalse(json.loads(self.output.read_text())["complete"])

    def test_existing_transcript_prevents_network_call(self) -> None:
        self.output.write_text("keep me", encoding="utf-8")
        with patch("examples.demos.chat.urlopen") as opened, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main([str(self.request), "--output", str(self.output)]), 1)
        opened.assert_not_called()
        self.assertEqual(self.output.read_text(), "keep me")

    def test_offline_preview_needs_no_credentials_or_network(self) -> None:
        for path in sorted((Path(__file__).parents[1] / "examples/demos/requests").glob("*.json")):
            with self.subTest(path=path), patch("examples.demos.chat.urlopen") as opened:
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    self.assertEqual(main([str(path), "--show-request", "--password-file", "/missing"]), 0)
                self.assertEqual(json.loads(out.getvalue()), json.loads(path.read_text()))
                opened.assert_not_called()

    def test_multiline_sse_and_unterminated_event(self) -> None:
        lines = [b'data: {"type": "content",\r\n', b'data: "delta": "test"}\r\n', b"\r\n"]
        self.assertEqual(list(events(lines)), [{"type": "content", "delta": "test"}])
        with self.assertRaisesRegex(ValueError, "inside an event"):
            list(events(lines[:-1]))


if __name__ == "__main__":
    unittest.main()
