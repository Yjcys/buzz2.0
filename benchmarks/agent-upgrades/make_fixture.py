#!/usr/bin/env python3
"""Generate a deterministic workspace and task files for acp_eval.py.

Two task sets:

- tool_use.jsonl: short questions that need one or two tool calls (line
  counts, config values). Exercises tool-call reliability.
- long_context.jsonl: each task makes the agent read ten ~16 KB log files in
  full and add up a checksum hidden in each. About 40k tokens of tool output
  per task, so with BUZZ_AGENT_MAX_CONTEXT_TOKENS=32000 every run has to
  reclaim context. A controlled stress test for context management.

Answers are computed here, so checks are exact.
"""

import argparse
import json
import random
from pathlib import Path

WORDS = (
    "agent relay channel thread token cache window summary tool result shell "
    "file patch review build test deploy queue session prompt model context"
).split()


def write_tool_use(ws: Path, rng: random.Random) -> list[dict]:
    d = ws / "tool_use"
    d.mkdir(parents=True, exist_ok=True)
    tasks = []
    for i in range(10):
        n = rng.randint(17, 140)
        lines = [" ".join(rng.choices(WORDS, k=8)) for _ in range(n)]
        name = f"notes_{i:02d}.txt"
        (d / name).write_text("\n".join(lines) + "\n")
        tasks.append(
            {
                "id": f"lines-{i:02d}",
                "prompt": f"How many lines are in tool_use/{name}? Use your tools to check. "
                "Reply with only the number.",
                "check": {"regex": rf"(?<!\d){n}(?!\d)"},
                "needs_tools": True,
            }
        )
    for i in range(10):
        port = rng.randint(2000, 9999)
        name = f"service_{i:02d}.ini"
        body = (
            f"[server]\nhost = svc-{i}.internal\nport = {port}\n"
            f"workers = {rng.randint(2, 16)}\n\n[log]\nlevel = info\n"
        )
        (d / name).write_text(body)
        tasks.append(
            {
                "id": f"port-{i:02d}",
                "prompt": f"What port does tool_use/{name} configure for the server? "
                "Reply with only the number.",
                "check": {"regex": rf"(?<!\d){port}(?!\d)"},
                "needs_tools": True,
            }
        )
    return tasks


def write_long_context(ws: Path, rng: random.Random) -> list[dict]:
    d = ws / "long"
    d.mkdir(parents=True, exist_ok=True)
    checksums = {}
    for i in range(40):
        value = rng.randint(1000, 9999)
        checksums[i] = value
        lines = []
        hidden_at = rng.randint(40, 160)
        for j in range(200):
            if j == hidden_at:
                lines.append(f"2026-10-12T10:{j % 60:02d}:00Z INFO checksum part_{i:02d} = {value}")
            else:
                filler = " ".join(rng.choices(WORDS, k=9))
                lines.append(f"2026-10-12T10:{j % 60:02d}:00Z DEBUG worker-{j % 7} {filler}")
        (d / f"part_{i:02d}.log").write_text("\n".join(lines) + "\n")
    tasks = []
    for t in range(10):
        picks = sorted(rng.sample(range(40), 10))
        files = ", ".join(f"long/part_{p:02d}.log" for p in picks)
        total = sum(checksums[p] for p in picks)
        tasks.append(
            {
                "id": f"sum-{t:02d}",
                "prompt": "This is a reading test. Read each of these files in full with your "
                f"file-reading tool, one at a time (do not use grep or rg): {files}. Each "
                "contains one line of the form `checksum part_NN = VALUE`. Reply with only "
                "the sum of the ten values.",
                "check": {"regex": rf"(?<!\d){total}(?!\d)"},
                "needs_tools": True,
            }
        )
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/tmp/buzz-agent-eval", help="output directory")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    ws = out / "ws"
    rng = random.Random(args.seed)
    sets = {
        "tool_use": write_tool_use(ws, rng),
        "long_context": write_long_context(ws, rng),
    }
    for name, tasks in sets.items():
        path = out / f"{name}.jsonl"
        with path.open("w") as f:
            for task in tasks:
                f.write(json.dumps({**task, "workspace": str(ws)}) + "\n")
        print(f"{path}: {len(tasks)} tasks")


if __name__ == "__main__":
    main()
