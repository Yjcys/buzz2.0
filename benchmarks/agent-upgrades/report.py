#!/usr/bin/env python3
"""Summarize acp_eval.py results per condition as a Markdown table.

    python3 report.py results.jsonl
"""

import json
import sys
from collections import defaultdict


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def fmt(value, pct=False):
    if value is None:
        return "-"
    return f"{value * 100:.1f}%" if pct else f"{value:,.0f}"


def main():
    rows = [json.loads(line) for path in sys.argv[1:] for line in open(path) if line.strip()]
    groups = defaultdict(list)
    for row in rows:
        groups[row["label"]].append(row)

    cols = [
        "condition",
        "runs",
        "pass rate",
        "silent end rate",
        "turn errors",
        "avg input tokens",
        "avg first-request input",
        "cache-read share",
        "format recoveries",
        "mask events",
        "handoffs",
    ]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for label, rs in groups.items():
        n = len(rs)
        # Silent end: the turn finished normally but no tool ever ran and the
        # answer failed the check — the #7982 symptom.
        silent = sum(
            1
            for r in rs
            if r["stop_reason"] == "end_turn" and r["tool_calls"] == 0 and not r["passed"]
        )
        inputs = [r["usage"].get("accumulatedInputTokens") for r in rs]
        cached = [r["usage"].get("accumulatedCachedInputTokens") for r in rs]
        pairs = [(c, i) for c, i in zip(cached, inputs) if c is not None and i]
        share = sum(c for c, _ in pairs) / sum(i for _, i in pairs) if pairs else None
        print(
            "| "
            + " | ".join(
                [
                    label,
                    str(n),
                    fmt(sum(r["passed"] for r in rs) / n, pct=True),
                    fmt(silent / n, pct=True),
                    str(sum(1 for r in rs if r["error"])),
                    fmt(mean(inputs)),
                    fmt(mean([r.get("first_request_input_tokens") for r in rs])),
                    fmt(share, pct=True),
                    str(sum(r.get("format_recoveries", 0) for r in rs)),
                    str(sum(r.get("mask_events", 0) for r in rs)),
                    str(sum(r.get("handoffs", 0) for r in rs)),
                ]
            )
            + " |"
        )


if __name__ == "__main__":
    main()
