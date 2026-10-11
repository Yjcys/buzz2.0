#!/usr/bin/env python3
"""Run tasks against buzz-agent over ACP stdio and record one JSON row each.

No relay, Docker, or Harbor: the script plays the ACP client. It starts
buzz-agent with buzz-dev-mcp as its only MCP server, approves every
permission request, collects tool calls, the final answer, and the
session's usage counters, and greps the agent's stderr for the events the
upgrades log (tool-format recovery, observation masking, handoffs).

Example (baseline vs upgrade on the same tasks):

    python3 acp_eval.py --tasks /tmp/buzz-agent-eval/tool_use.jsonl \\
        --label baseline --env BUZZ_AGENT_MAX_TOOL_FORMAT_RECOVERIES=0 --out results.jsonl
    python3 acp_eval.py --tasks /tmp/buzz-agent-eval/tool_use.jsonl \\
        --label upgrade --out results.jsonl

Model settings (BUZZ_AGENT_PROVIDER, OPENAI_COMPAT_*, ...) come from your
environment or --env.
"""

import argparse
import json
import os
import queue
import re
import subprocess
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# stderr markers written by buzz-agent, counted per run.
MARKERS = {
    "format_recoveries": "tool call not issued in structured form",
    "format_exhausted": "recovery budget exhausted, ending turn",
    "tool_use_without_calls_error": "stop=tool_use but zero tool_calls",
    "mask_events": "observation mask: cleared",
    "handoffs": "handoff #",
}


class Agent:
    def __init__(self, cmd, env, cwd, stderr_path):
        self.stderr = open(stderr_path, "w")
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            env=env,
            cwd=cwd,
            text=True,
            bufsize=1,
        )
        self.lines: "queue.Queue[str | None]" = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self.next_id = 1

    def _pump(self):
        for line in self.proc.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def send(self, msg):
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def request(self, method, params):
        rid = self.next_id
        self.next_id += 1
        self.send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        return rid

    def recv(self, deadline):
        timeout = deadline - time.monotonic()
        if timeout <= 0:
            raise TimeoutError
        line = self.lines.get(timeout=timeout)
        if line is None:
            raise EOFError("agent exited")
        return json.loads(line)

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
        self.stderr.close()


def approve(msg):
    options = msg["params"].get("options", [])
    option = next((o for o in options if o.get("kind") == "allow_once"), None)
    if option is None:
        return {"jsonrpc": "2.0", "id": msg["id"], "result": {"outcome": {"outcome": "cancelled"}}}
    return {
        "jsonrpc": "2.0",
        "id": msg["id"],
        "result": {"outcome": {"outcome": "selected", "optionId": option["optionId"]}},
    }


def run_task(task, args, env, log_path):
    workspace = task.get("workspace") or args.workspace
    agent = Agent([args.agent], env, workspace, log_path)
    row = {
        "label": args.label,
        "task": task["id"],
        "stop_reason": None,
        "error": None,
        "tool_calls": 0,
        "answer": "",
        "usage": {},
    }
    started = time.monotonic()
    deadline = started + args.timeout
    try:
        rid = agent.request("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
        while agent.recv(deadline).get("id") != rid:
            pass
        rid = agent.request(
            "session/new",
            {
                "cwd": workspace,
                "mcpServers": [{"name": "dev", "command": args.mcp, "args": [], "env": []}],
            },
        )
        while True:
            msg = agent.recv(deadline)
            if msg.get("id") == rid:
                if "error" in msg:
                    raise RuntimeError(f"session/new failed: {msg['error']}")
                sid = msg["result"]["sessionId"]
                break
        rid = agent.request(
            "session/prompt",
            {"sessionId": sid, "prompt": [{"type": "text", "text": task["prompt"]}]},
        )
        last_text = ""
        while True:
            msg = agent.recv(deadline)
            method = msg.get("method")
            if method == "session/request_permission":
                agent.send(approve(msg))
                continue
            update = msg.get("params", {}).get("update", {})
            kind = update.get("sessionUpdate")
            if method == "session/update" and kind == "tool_call":
                row["tool_calls"] += 1
            elif method == "session/update" and kind == "agent_message_chunk":
                text = update.get("content", {}).get("text", "")
                if text.strip():
                    last_text = text
            elif kind == "usage_update":
                # The first report covers exactly one request: system prompt,
                # tool definitions, and the task. Comparing it across prompt
                # profiles isolates the prompt's size in provider tokens.
                if "first_request_input_tokens" not in row:
                    row["first_request_input_tokens"] = update.get("accumulatedInputTokens")
                row["usage"] = {
                    k: update[k]
                    for k in (
                        "accumulatedInputTokens",
                        "accumulatedOutputTokens",
                        "accumulatedCachedInputTokens",
                        "accumulatedCacheWriteTokens",
                    )
                    if k in update
                }
            if msg.get("id") == rid:
                if "error" in msg:
                    row["error"] = msg["error"].get("message", str(msg["error"]))
                else:
                    row["stop_reason"] = msg["result"].get("stopReason")
                break
        row["answer"] = last_text
    except TimeoutError:
        row["error"] = "timeout"
    except Exception as e:  # noqa: BLE001 - record and continue with the next task
        row["error"] = f"{type(e).__name__}: {e}"
    finally:
        agent.close()
    row["seconds"] = round(time.monotonic() - started, 1)
    stderr = Path(log_path).read_text(errors="replace")
    for key, needle in MARKERS.items():
        row[key] = stderr.count(needle)
    check = task.get("check", {})
    # A task whose answer depends on file contents cannot be answered without
    # a tool, so an answer that only quotes an unexecuted call does not pass.
    row["passed"] = bool(
        row["error"] is None
        and "regex" in check
        and re.search(check["regex"], row["answer"])
        and (row["tool_calls"] > 0 or not task.get("needs_tools", False))
    )
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True, help="JSONL from make_fixture.py")
    ap.add_argument("--label", required=True, help="condition name, e.g. baseline / upgrade")
    ap.add_argument("--out", required=True, help="JSONL to append results to")
    ap.add_argument("--agent", default=str(REPO / "target/debug/buzz-agent"))
    ap.add_argument("--mcp", default=str(REPO / "target/debug/buzz-dev-mcp"))
    ap.add_argument("--workspace", help="cwd when a task has no workspace field")
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE for buzz-agent")
    ap.add_argument("--repeat", type=int, default=1, help="attempts per task")
    ap.add_argument("--timeout", type=float, default=600, help="seconds per attempt")
    ap.add_argument("--only", help="regex on task id")
    ap.add_argument("--logs", default="eval-logs", help="directory for per-run stderr")
    args = ap.parse_args()

    env = dict(os.environ)
    for item in args.env:
        key, _, value = item.partition("=")
        env[key] = value
    tasks = [json.loads(line) for line in open(args.tasks) if line.strip()]
    if args.only:
        tasks = [t for t in tasks if re.search(args.only, t["id"])]
    logs = Path(args.logs)
    logs.mkdir(parents=True, exist_ok=True)

    with open(args.out, "a") as out:
        for attempt in range(args.repeat):
            for task in tasks:
                log = logs / f"{args.label}-{task['id']}-{attempt}.log"
                row = run_task(task, args, env, log)
                row["attempt"] = attempt
                row["env"] = args.env
                out.write(json.dumps(row) + "\n")
                out.flush()
                status = "PASS" if row["passed"] else "FAIL"
                print(
                    f"[{args.label}] {task['id']}#{attempt} {status} stop={row['stop_reason']} "
                    f"tools={row['tool_calls']} in={row['usage'].get('accumulatedInputTokens')} "
                    f"err={row['error']}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
