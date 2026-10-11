# agent-upgrades evaluation kit

Measures four agent changes against their baselines:

| Change | Switch (baseline value) |
|---|---|
| Tool calls written as text, or `stop=tool_use` with no calls, are reported to the model and re-issued instead of ending or failing the turn | `BUZZ_AGENT_MAX_TOOL_FORMAT_RECOVERIES=0` |
| Older tool output is masked at 60% of the window before any handoff | `BUZZ_AGENT_OBSERVATION_MASK_PCT=0` |
| `chat` base-prompt profile | `BUZZ_ACP_PROMPT_PROFILE=full` |
| Failure notices are tagged and never wake sibling agents | build from `main` |

Files:

- `make_fixture.py` writes a deterministic workspace and two task sets (`tool_use`, `long_context`) with exact answers.
- `acp_eval.py` drives `buzz-agent` over ACP stdio with `buzz-dev-mcp` as its tool server. No relay or Docker. One JSON row per run: pass/fail, stop reason, tool calls, session token counters, and counts of the agent's recovery, masking, and handoff log lines.
- `report.py` prints a Markdown table per condition.
- `prompt_tokens.py` sizes the base prompt under both profiles and can emit both as files.
- `mock_llm.py` + `mock_tasks.jsonl` run the whole pipeline against a scripted model for free.
- `endpoints/` holds Harbor endpoint configs that set the prompt profile for `just benchmark`.

Dry run (no API key needed):

```bash
cargo build -p buzz-agent -p buzz-dev-mcp
python3 mock_llm.py --port 8765 &
export BUZZ_AGENT_PROVIDER=openai OPENAI_COMPAT_API_KEY=x OPENAI_COMPAT_MODEL=mock \
  OPENAI_COMPAT_BASE_URL=http://127.0.0.1:8765/v1 OPENAI_COMPAT_API=chat
python3 acp_eval.py --tasks mock_tasks.jsonl --label baseline \
  --env BUZZ_AGENT_MAX_TOOL_FORMAT_RECOVERIES=0 --out mock.jsonl
python3 acp_eval.py --tasks mock_tasks.jsonl --label upgrade --out mock.jsonl
python3 report.py mock.jsonl
```

The baseline ends the turn without running the shell (silent end); the upgrade recovers and passes.
