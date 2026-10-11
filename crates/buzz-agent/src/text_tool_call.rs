//! Detection of tool calls a model wrote into its reply text instead of
//! issuing them through the provider's structured tool-calling interface.
//!
//! Some served models (commonly open-weight models behind OpenAI-compatible
//! endpoints without a matching tool parser) emit their chat template's call
//! markup, or a bare JSON call, as ordinary assistant text. The provider then
//! reports `end_turn` with no tool calls, so the turn ends without the call
//! ever running and without any error ([#7982]).
//!
//! This module only detects. A detected call is never executed: text that
//! looks like a call has passed none of the checks a structured call has, and
//! the model may simply be quoting one. The caller tells the model the call
//! did not run and lets it re-issue the call properly.
//!
//! To keep ordinary prose from matching, a detection requires a registered
//! tool name, plus either a known call marker in the text or a reply that is
//! nothing but a call-shaped JSON value.
//!
//! [#7982]: https://github.com/block/buzz/issues/7982

use serde_json::Value;

use crate::mcp::truncate_at_boundary;
use crate::types::ToolDef;

/// Chat-template call markers seen in the wild (Hermes/Qwen, Mistral,
/// Llama 3.x, DeepSeek, Llama 4/Qwen3-Coder XML, Anthropic-style XML).
const CALL_MARKERS: &[&str] = &[
    "<tool_call>",
    "[TOOL_CALLS]",
    "<|python_tag|>",
    "<｜tool▁call",
    "<|tool_call",
    "<function=",
    "<invoke name=",
    "<tool_use>",
];

/// Argument keys that, next to `name`, mark an object as a call.
const ARGUMENT_KEYS: &[&str] = &["arguments", "parameters", "input", "args"];

/// Only the head of a long reply is scanned; a call buried deeper than this
/// in prose is not worth the parsing cost.
const MAX_SCAN_BYTES: usize = 64 * 1024;
/// Upper bound on JSON parse attempts per reply.
const MAX_JSON_STARTS: usize = 256;
/// Per-attempt input bound; real calls are far smaller.
const MAX_JSON_BYTES: usize = 16 * 1024;

/// Return the registered tool name of a call written as text, if any.
pub(crate) fn detect(text: &str, tools: &[ToolDef]) -> Option<String> {
    let text = truncate_at_boundary(text, MAX_SCAN_BYTES).trim();
    if text.is_empty() || tools.is_empty() {
        return None;
    }
    if let Some(name) = markup_call_name(text, tools) {
        return Some(name);
    }
    if CALL_MARKERS.iter().any(|m| text.contains(m)) {
        return json_starts(text)
            .filter_map(|start| parse_prefix(&text[start..]))
            .find_map(|value| call_in_value(&value, tools));
    }
    // No marker: only a reply that is entirely a call counts.
    whole_json(strip_code_fence(text)).and_then(|value| call_in_value(&value, tools))
}

/// `<function=NAME>` and `<invoke name="NAME">` carry the name in the markup
/// itself, with XML-style parameters that are not JSON.
fn markup_call_name(text: &str, tools: &[ToolDef]) -> Option<String> {
    for (marker, terminators) in [
        ("<function=", &['>', ' ', '\n'][..]),
        ("<invoke name=", &['>', ' ', '\n', '"', '\''][..]),
    ] {
        for (idx, _) in text.match_indices(marker) {
            let rest = text[idx + marker.len()..].trim_start_matches(['"', '\'']);
            let name = rest.split(terminators).next().unwrap_or("");
            if let Some(found) = registered(name, tools) {
                return Some(found);
            }
        }
    }
    None
}

fn json_starts(text: &str) -> impl Iterator<Item = usize> + '_ {
    text.match_indices('{')
        .map(|(i, _)| i)
        .take(MAX_JSON_STARTS)
}

/// Parse the first JSON value at the start of `s`, ignoring what follows.
fn parse_prefix(s: &str) -> Option<Value> {
    let s = truncate_at_boundary(s, MAX_JSON_BYTES);
    serde_json::Deserializer::from_str(s)
        .into_iter::<Value>()
        .next()?
        .ok()
}

/// Parse `s` only if it is exactly one JSON value.
fn whole_json(s: &str) -> Option<Value> {
    if !(s.starts_with('{') || s.starts_with('[')) || s.len() > MAX_JSON_BYTES {
        return None;
    }
    serde_json::from_str(s).ok()
}

/// Strip one surrounding Markdown code fence, if present.
fn strip_code_fence(text: &str) -> &str {
    let Some(body) = text.strip_prefix("```") else {
        return text;
    };
    let Some(body) = body.strip_suffix("```") else {
        return text;
    };
    // Drop the info string (e.g. `json`) on the opening line.
    body.split_once('\n').map_or(body, |(_, rest)| rest).trim()
}

fn call_in_value(value: &Value, tools: &[ToolDef]) -> Option<String> {
    match value {
        Value::Array(items) => items.iter().find_map(|v| call_in_value(v, tools)),
        Value::Object(obj) => {
            if let Some(function) = obj.get("function").and_then(Value::as_object) {
                if let Some(name) = function.get("name").and_then(Value::as_str) {
                    return registered(name, tools);
                }
            }
            let has_args = ARGUMENT_KEYS.iter().any(|k| obj.contains_key(*k));
            let name = obj
                .get("name")
                .or_else(|| obj.get("tool"))
                .and_then(Value::as_str);
            match name {
                Some(name) if has_args => registered(name, tools),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Resolve `name` against the tool catalog: exact match first, then a bare
/// tool name against the `server__tool` qualified form.
fn registered(name: &str, tools: &[ToolDef]) -> Option<String> {
    let name = name.trim();
    if name.is_empty() {
        return None;
    }
    if let Some(t) = tools.iter().find(|t| t.name == name) {
        return Some(t.name.clone());
    }
    let suffix = format!("__{name}");
    tools
        .iter()
        .find(|t| t.name.ends_with(&suffix))
        .map(|t| t.name.clone())
}

#[cfg(test)]
mod tests {
    use super::detect;
    use crate::types::ToolDef;
    use serde_json::json;

    fn tools() -> Vec<ToolDef> {
        ["dev__shell", "dev__str_replace", "load_skill"]
            .into_iter()
            .map(|name| ToolDef {
                name: name.into(),
                description: String::new(),
                input_schema: json!({"type": "object"}),
            })
            .collect()
    }

    fn found(text: &str) -> Option<String> {
        detect(text, &tools())
    }

    #[test]
    fn hermes_tool_call_tags() {
        let text = "Let me check.\n<tool_call>\n{\"name\": \"dev__shell\", \"arguments\": {\"command\": \"ls\"}}\n</tool_call>";
        assert_eq!(found(text).as_deref(), Some("dev__shell"));
    }

    #[test]
    fn bare_tool_name_resolves_to_qualified_name() {
        let text =
            "<tool_call>{\"name\": \"shell\", \"arguments\": {\"command\": \"ls\"}}</tool_call>";
        assert_eq!(found(text).as_deref(), Some("dev__shell"));
    }

    #[test]
    fn mistral_array_after_marker() {
        let text =
            "[TOOL_CALLS] [{\"name\": \"dev__str_replace\", \"arguments\": {\"path\": \"a\"}}]";
        assert_eq!(found(text).as_deref(), Some("dev__str_replace"));
    }

    #[test]
    fn llama_python_tag_with_parameters() {
        let text =
            "<|python_tag|>{\"name\": \"dev__shell\", \"parameters\": {\"command\": \"pwd\"}}";
        assert_eq!(found(text).as_deref(), Some("dev__shell"));
    }

    #[test]
    fn xml_function_markup() {
        let text = "<function=dev__shell>\n<parameter=command>ls</parameter>\n</function>";
        assert_eq!(found(text).as_deref(), Some("dev__shell"));
        let text = "<invoke name=\"shell\"><parameter name=\"command\">ls</parameter></invoke>";
        assert_eq!(found(text).as_deref(), Some("dev__shell"));
    }

    #[test]
    fn whole_reply_json_call() {
        let text = "{\"name\": \"dev__shell\", \"arguments\": {\"command\": \"ls\"}}";
        assert_eq!(found(text).as_deref(), Some("dev__shell"));
    }

    #[test]
    fn fenced_openai_function_shape() {
        let text = "```json\n{\"type\": \"function\", \"function\": {\"name\": \"load_skill\", \"arguments\": \"{}\"}}\n```";
        assert_eq!(found(text).as_deref(), Some("load_skill"));
    }

    #[test]
    fn prose_mentioning_a_tool_is_not_a_call() {
        assert_eq!(found("I ran dev__shell and the tests pass."), None);
        assert_eq!(found("Use the shell tool with arguments like `ls`."), None);
    }

    #[test]
    fn json_example_inside_prose_without_marker_is_not_a_call() {
        let text = "The request body looks like {\"name\": \"dev__shell\", \"arguments\": {}} in the docs.";
        assert_eq!(found(text), None);
    }

    #[test]
    fn unregistered_tool_name_is_not_a_call() {
        let text =
            "<tool_call>{\"name\": \"web_search\", \"arguments\": {\"q\": \"x\"}}</tool_call>";
        assert_eq!(found(text), None);
        assert_eq!(found("<function=web_search>q</function>"), None);
    }

    #[test]
    fn json_without_arguments_key_is_not_a_call() {
        assert_eq!(found("{\"name\": \"dev__shell\"}"), None);
    }

    #[test]
    fn marker_without_a_call_is_not_a_call() {
        assert_eq!(found("The template wraps calls in <tool_call> tags."), None);
    }

    #[test]
    fn empty_catalog_never_matches() {
        let text = "<tool_call>{\"name\": \"dev__shell\", \"arguments\": {}}</tool_call>";
        assert_eq!(detect(text, &[]), None);
    }
}
