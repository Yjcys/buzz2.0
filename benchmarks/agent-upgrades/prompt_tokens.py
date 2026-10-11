#!/usr/bin/env python3
"""Size of the base prompt under the full and chat profiles.

Splits crates/buzz-acp/src/base_prompt.md the same way
crates/buzz-acp/src/prompt_profile.rs does (keep this list in sync) and
prints bytes and, if tiktoken is installed, o200k_base token counts. With
--emit DIR it also writes full.md and chat.md, which can be passed to
buzz-agent as BUZZ_AGENT_SYSTEM_PROMPT_FILE to measure provider-reported
input tokens directly (see the evaluation plan).
"""

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ACP = REPO / "crates/buzz-acp/src"
ON_DEMAND = {
    "Projects",
    "Conversational Agent Creation",
    "Workspace Layout",
    "Engineering Discipline",
    "Working in the Repo",
}
SKILL = "/path/to/.agents/skills/buzz-workflows/SKILL.md"


def chat_prompt(base: str) -> str:
    kept, moving = [], False
    for line in base.splitlines(keepends=True):
        if line.startswith("## "):
            moving = line[3:].strip() in ON_DEMAND
        if not moving:
            kept.append(line)
    return (
        "".join(kept).rstrip()
        + "\n\n## On-Demand Guidance\n\nThis agent runs a lean base prompt. Before working on "
        "projects, repositories, pull requests, issues, creating agents, or code in a repo, read "
        f"`{SKILL}` and follow it. It also describes your workspace layout.\n"
    )


def count_tokens(text: str):
    try:
        import tiktoken
    except ImportError:
        return None
    return len(tiktoken.get_encoding("o200k_base").encode(text))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emit", help="directory to write full.md and chat.md")
    args = ap.parse_args()
    session_model = (ACP / "session_model_channel.md").read_text()
    base = (ACP / "base_prompt.md").read_text()
    variants = {
        "full": base.rstrip() + "\n\n" + session_model,
        "chat": chat_prompt(base).rstrip() + "\n\n" + session_model,
    }
    sizes = {}
    for name, text in variants.items():
        sizes[name] = (len(text.encode()), count_tokens(text))
        tokens = sizes[name][1]
        print(f"{name:5s} {sizes[name][0]:6d} bytes  {tokens if tokens is not None else '?':>6} tokens")
    full_b, chat_b = sizes["full"][0], sizes["chat"][0]
    print(f"reduction {100 * (1 - chat_b / full_b):.1f}% bytes", end="")
    if sizes["full"][1]:
        print(f", {100 * (1 - sizes['chat'][1] / sizes['full'][1]):.1f}% tokens")
    else:
        print(" (pip install tiktoken for token counts)")
    if args.emit:
        out = Path(args.emit)
        out.mkdir(parents=True, exist_ok=True)
        for name, text in variants.items():
            (out / f"{name}.md").write_text(text)
        print(f"wrote {out}/full.md and {out}/chat.md")


if __name__ == "__main__":
    main()
