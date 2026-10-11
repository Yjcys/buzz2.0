//! Observation masking: the cheap first rung of context reclamation.
//!
//! Long agent turns are dominated by old tool output (file reads, command
//! logs) that the model has already acted on. Before paying for an LLM
//! summary and a full history reset (the handoff), replace the content of
//! older tool results with a one-line placeholder that keeps the tool name and
//! arguments. The model keeps its record of what it did and can re-run a tool
//! when it needs the output again.
//!
//! Invariants:
//! - Only tool-result *content* changes. Every result keeps its `provider_id`
//!   and position, so `tool_use` ↔ `tool_result` pairing stays valid on every
//!   provider wire format, and assistant turns (including replayed reasoning)
//!   are untouched.
//! - The newest `keep_rounds` tool rounds are never masked.
//! - Error results and `load_skill` results are never masked: errors are cheap
//!   and steer the next attempt, and skill bodies are instructions.
//! - Masking happens in batches: nothing changes unless at least
//!   `min_freed_bytes` can be freed. Every edit invalidates the provider's
//!   prompt cache from the edited item onward, so many small edits would cost
//!   more than they save.

use std::collections::HashMap;

use crate::builtin::LOAD_SKILL_TOOL;
use crate::mcp::truncate_at_boundary;
use crate::types::{HistoryItem, ToolResultContent};

/// Results smaller than this are not worth a placeholder.
pub(crate) const MIN_MASKABLE_RESULT_BYTES: usize = 1024;
/// Minimum total to free before any edit is made (see module docs).
pub(crate) const MIN_FREED_BYTES: usize = 8 * 1024;
/// Prefix that marks an already-masked result.
const MASK_MARKER: &str = "[Cleared to save context]";
/// Cap on the argument summary kept in a placeholder.
const MAX_ARGS_BYTES: usize = 160;

#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct MaskStats {
    pub masked: usize,
    /// Context-pressure bytes removed, net of the placeholders added.
    pub freed_bytes: usize,
}

/// Mask tool results older than the newest `keep_rounds` tool rounds, if doing
/// so frees at least `min_freed_bytes`. Returns what was masked.
pub(crate) fn mask_old_tool_results(
    history: &mut [HistoryItem],
    keep_rounds: usize,
    min_freed_bytes: usize,
) -> MaskStats {
    let tool_rounds: Vec<usize> = history
        .iter()
        .enumerate()
        .filter(|(_, item)| {
            matches!(item, HistoryItem::Assistant { tool_calls, .. } if !tool_calls.is_empty())
        })
        .map(|(i, _)| i)
        .collect();
    if tool_rounds.len() <= keep_rounds {
        return MaskStats::default();
    }
    let boundary = tool_rounds[tool_rounds.len() - keep_rounds.max(1)];

    let mut calls: HashMap<&str, (&str, String)> = HashMap::new();
    for item in &history[..boundary] {
        if let HistoryItem::Assistant { tool_calls, .. } = item {
            for c in tool_calls {
                let args = c.arguments.to_string();
                let args = truncate_at_boundary(&args, MAX_ARGS_BYTES).to_owned();
                calls.insert(c.provider_id.as_str(), (c.name.as_str(), args));
            }
        }
    }

    // Plan first so a batch below the floor makes no edit at all.
    let mut plan: Vec<(usize, String)> = Vec::new();
    let mut freed = 0usize;
    for (i, item) in history[..boundary].iter().enumerate() {
        let HistoryItem::ToolResult(result) = item else {
            continue;
        };
        if result.is_error || is_masked(&result.content) {
            continue;
        }
        let (name, args) = calls
            .get(result.provider_id.as_str())
            .map(|(n, a)| (*n, a.as_str()))
            .unwrap_or(("tool", "?"));
        if name == LOAD_SKILL_TOOL {
            continue;
        }
        let size: usize = result
            .content
            .iter()
            .map(ToolResultContent::context_pressure_bytes)
            .sum();
        if size < MIN_MASKABLE_RESULT_BYTES {
            continue;
        }
        let placeholder = format!(
            "{MASK_MARKER} Output of {name} {args} ({size} bytes) was removed. \
             Run the tool again if you need it."
        );
        freed += size.saturating_sub(placeholder.len());
        plan.push((i, placeholder));
    }
    if plan.is_empty() || freed < min_freed_bytes {
        return MaskStats::default();
    }

    let masked = plan.len();
    let history = &mut history[..boundary];
    for (i, placeholder) in plan {
        if let HistoryItem::ToolResult(result) = &mut history[i] {
            result.content = vec![ToolResultContent::Text(placeholder)];
        }
    }
    MaskStats {
        masked,
        freed_bytes: freed,
    }
}

fn is_masked(content: &[ToolResultContent]) -> bool {
    matches!(content, [ToolResultContent::Text(t)] if t.starts_with(MASK_MARKER))
}

#[cfg(test)]
mod tests {
    use super::{mask_old_tool_results, MaskStats, MASK_MARKER};
    use crate::types::{HistoryItem, ToolCall, ToolResult, ToolResultContent};
    use serde_json::json;

    fn call(id: &str, name: &str) -> ToolCall {
        ToolCall {
            provider_id: id.into(),
            name: name.into(),
            arguments: json!({"path": format!("/src/{id}.rs")}),
            provider_extra: Default::default(),
        }
    }

    fn round(history: &mut Vec<HistoryItem>, id: &str, name: &str, bytes: usize, is_error: bool) {
        history.push(HistoryItem::Assistant {
            text: String::new(),
            tool_calls: vec![call(id, name)],
            reasoning_details: None,
        });
        history.push(HistoryItem::ToolResult(ToolResult {
            provider_id: id.into(),
            content: vec![ToolResultContent::Text("x".repeat(bytes))],
            is_error,
        }));
    }

    fn text_of(item: &HistoryItem) -> String {
        match item {
            HistoryItem::ToolResult(r) => r.text(),
            _ => panic!("not a tool result"),
        }
    }

    #[test]
    fn masks_old_results_and_keeps_recent_rounds() {
        let mut h = vec![HistoryItem::User("task".into())];
        for id in ["a", "b", "c", "d"] {
            round(&mut h, id, "dev__read_file", 10_000, false);
        }
        let stats = mask_old_tool_results(&mut h, 2, 1);
        assert_eq!(stats.masked, 2);
        assert!(stats.freed_bytes > 19_000);
        assert!(text_of(&h[2]).starts_with(MASK_MARKER));
        assert!(text_of(&h[2]).contains("dev__read_file"));
        assert!(text_of(&h[2]).contains("/src/a.rs"));
        assert!(text_of(&h[4]).starts_with(MASK_MARKER));
        assert_eq!(text_of(&h[6]).len(), 10_000, "kept round was masked");
        assert_eq!(text_of(&h[8]).len(), 10_000, "kept round was masked");
    }

    #[test]
    fn pairing_ids_survive_masking() {
        let mut h = Vec::new();
        for id in ["a", "b"] {
            round(&mut h, id, "dev__shell", 5_000, false);
        }
        mask_old_tool_results(&mut h, 1, 1);
        match &h[1] {
            HistoryItem::ToolResult(r) => assert_eq!(r.provider_id, "a"),
            _ => panic!("result moved"),
        }
    }

    #[test]
    fn below_the_batch_floor_nothing_changes() {
        let mut h = Vec::new();
        for id in ["a", "b", "c"] {
            round(&mut h, id, "dev__shell", 2_000, false);
        }
        let before = h.clone();
        let stats = mask_old_tool_results(&mut h, 1, 8 * 1024);
        assert_eq!(stats, MaskStats::default());
        assert_eq!(format!("{before:?}"), format!("{h:?}"));
    }

    #[test]
    fn errors_skills_and_small_results_are_kept() {
        let mut h = Vec::new();
        round(&mut h, "err", "dev__shell", 5_000, true);
        round(&mut h, "skill", "load_skill", 5_000, false);
        round(&mut h, "tiny", "dev__shell", 100, false);
        round(&mut h, "big", "dev__shell", 5_000, false);
        round(&mut h, "recent", "dev__shell", 5_000, false);
        let stats = mask_old_tool_results(&mut h, 1, 1);
        assert_eq!(stats.masked, 1);
        assert!(!text_of(&h[1]).starts_with(MASK_MARKER));
        assert!(!text_of(&h[3]).starts_with(MASK_MARKER));
        assert!(!text_of(&h[5]).starts_with(MASK_MARKER));
        assert!(text_of(&h[7]).starts_with(MASK_MARKER));
    }

    #[test]
    fn masking_is_idempotent() {
        let mut h = Vec::new();
        for id in ["a", "b", "c"] {
            round(&mut h, id, "dev__shell", 5_000, false);
        }
        assert_eq!(mask_old_tool_results(&mut h, 1, 1).masked, 2);
        assert_eq!(mask_old_tool_results(&mut h, 1, 1), MaskStats::default());
    }

    #[test]
    fn too_few_rounds_is_a_noop() {
        let mut h = Vec::new();
        round(&mut h, "a", "dev__shell", 50_000, false);
        assert_eq!(mask_old_tool_results(&mut h, 3, 1), MaskStats::default());
    }
}
