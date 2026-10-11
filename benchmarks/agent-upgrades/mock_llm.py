#!/usr/bin/env python3
"""A scripted OpenAI-compatible server for a free dry run of acp_eval.py.

For every new conversation it replies, in order: a tool call written as plain
text (the #7982 failure), a real `dev__shell` call, then a final answer that
echoes the number the shell printed. It checks the whole pipeline (ACP
handshake, permission approval, usage parsing, stderr markers, report)
without spending tokens.

    python3 mock_llm.py --port 8765 &
    BUZZ_AGENT_PROVIDER=openai OPENAI_COMPAT_API_KEY=x OPENAI_COMPAT_MODEL=mock \\
      OPENAI_COMPAT_BASE_URL=http://127.0.0.1:8765/v1 OPENAI_COMPAT_API=chat \\
      python3 acp_eval.py --tasks mock_tasks.jsonl --label mock --out mock.jsonl
"""

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def reply(message, finish, prompt_tokens):
    return {
        "id": "mock",
        "object": "chat.completion",
        "model": "mock",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 10,
            "total_tokens": prompt_tokens + 10,
            "prompt_tokens_details": {"cached_tokens": prompt_tokens // 2},
        },
    }


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        messages = body.get("messages", [])
        tool_results = [m for m in messages if m.get("role") == "tool"]
        nudged = any(
            "was NOT executed" in str(m.get("content")) for m in messages if m.get("role") == "user"
        )
        size = len(json.dumps(messages)) // 4
        if tool_results:
            text = str(tool_results[-1].get("content"))
            number = (re.findall(r"\d+", text) or ["0"])[-1]
            out = reply({"role": "assistant", "content": number}, "stop", size)
        elif nudged:
            call = {
                "id": "call-1",
                "type": "function",
                "function": {"name": "dev__shell", "arguments": json.dumps({"command": "echo 42"})},
            }
            out = reply({"role": "assistant", "content": None, "tool_calls": [call]}, "tool_calls", size)
        else:
            text = '<tool_call>{"name": "dev__shell", "arguments": {"command": "echo 42"}}</tool_call>'
            out = reply({"role": "assistant", "content": text}, "stop", size)
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
