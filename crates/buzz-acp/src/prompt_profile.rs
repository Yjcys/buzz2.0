//! Base-prompt profiles.
//!
//! The compiled-in base prompt is written for coding agents, and every session
//! carries all of it. Sections about projects, repositories, agent creation and
//! engineering practice cost every request their tokens even for agents that
//! only answer questions, and on small models they crowd out the rules that
//! matter for every agent, such as publishing a reply (#7958).
//!
//! The `chat` profile keeps the sections every agent needs in the prompt and
//! moves the coding-only sections into an on-demand skill written to
//! `<cwd>/.agents/skills/buzz-workflows/SKILL.md`. The prompt names that file,
//! so any runtime with a file tool can load it when a task needs it; runtimes
//! that index `.agents/skills` also list it as a skill. `full` (the default)
//! leaves the base prompt unchanged.

use std::path::{Path, PathBuf};

/// Selects how much of the base prompt is sent with every session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum PromptProfile {
    /// The whole base prompt (coding agents).
    #[default]
    Full,
    /// Core sections only; coding and project guidance loads on demand.
    Chat,
}

impl std::fmt::Display for PromptProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Full => f.write_str("full"),
            Self::Chat => f.write_str("chat"),
        }
    }
}

/// Level-2 sections of the base prompt that only coding and project work
/// needs. Everything else stays in the prompt.
const ON_DEMAND_SECTIONS: &[&str] = &[
    "Projects",
    "Conversational Agent Creation",
    "Workspace Layout",
    "Engineering Discipline",
    "Working in the Repo",
];

const SKILL_NAME: &str = "buzz-workflows";
const SKILL_DESCRIPTION: &str = "Buzz guidance for projects, repositories, pull requests and \
    issues, creating agents, the workspace layout, and engineering work. Load before doing any \
    of these.";

/// The prompt and skill produced by the chat profile.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ChatBase {
    pub prompt: String,
    pub skill_md: String,
}

/// Resolve the base prompt for `profile`. For `chat`, installs the on-demand
/// skill under `cwd`; if that fails, falls back to the unchanged base so no
/// guidance is lost.
pub(crate) fn apply(profile: PromptProfile, base: &str, cwd: &Path) -> String {
    if profile == PromptProfile::Full {
        return base.to_owned();
    }
    let skill_path = skill_path(cwd);
    let Some(chat) = chat_base(base, &skill_path) else {
        return base.to_owned();
    };
    match install_skill(&skill_path, &chat.skill_md) {
        Ok(()) => {
            tracing::info!(
                full_bytes = base.len(),
                chat_bytes = chat.prompt.len(),
                skill = %skill_path.display(),
                "chat prompt profile: moved coding guidance into an on-demand skill"
            );
            chat.prompt
        }
        Err(e) => {
            tracing::warn!(
                skill = %skill_path.display(),
                "chat prompt profile: could not write skill ({e}); using the full base prompt"
            );
            base.to_owned()
        }
    }
}

fn skill_path(cwd: &Path) -> PathBuf {
    cwd.join(".agents/skills").join(SKILL_NAME).join("SKILL.md")
}

/// Split `base` at level-2 headings and move [`ON_DEMAND_SECTIONS`] into a
/// skill. Returns `None` when no such section exists (e.g. a custom base
/// file), in which case there is nothing to move.
pub(crate) fn chat_base(base: &str, skill_path: &Path) -> Option<ChatBase> {
    let mut kept = String::new();
    let mut moved = String::new();
    let mut moving = false;
    for line in base.split_inclusive('\n') {
        if let Some(title) = line.strip_prefix("## ") {
            moving = ON_DEMAND_SECTIONS.contains(&title.trim());
        }
        if moving {
            moved.push_str(line);
        } else {
            kept.push_str(line);
        }
    }
    if moved.is_empty() {
        return None;
    }
    let prompt = format!(
        "{}\n\n## On-Demand Guidance\n\nThis agent runs a lean base prompt. Before working on \
         projects, repositories, pull requests, issues, creating agents, or code in a repo, read \
         `{}` and follow it. It also describes your workspace layout.\n",
        kept.trim_end(),
        skill_path.display()
    );
    let skill_md = format!(
        "---\nname: {SKILL_NAME}\ndescription: {SKILL_DESCRIPTION}\n---\n\n\
         <!-- Written by buzz-acp for the chat prompt profile; regenerated on start. -->\n\n{}\n",
        moved.trim_end()
    );
    Some(ChatBase { prompt, skill_md })
}

/// Write the skill file, skipping the write when it is already current.
fn install_skill(path: &Path, contents: &str) -> std::io::Result<()> {
    if std::fs::read_to_string(path).is_ok_and(|current| current == contents) {
        return Ok(());
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, contents)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: &str = include_str!("base_prompt.md");

    fn headings(text: &str) -> Vec<&str> {
        text.lines()
            .filter_map(|l| l.strip_prefix("## "))
            .map(str::trim)
            .collect()
    }

    #[test]
    fn full_profile_is_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(apply(PromptProfile::Full, BASE, dir.path()), BASE);
        assert!(!skill_path(dir.path()).exists());
    }

    #[test]
    fn chat_profile_moves_every_on_demand_section_and_keeps_the_rest() {
        let chat = chat_base(BASE, Path::new("/nest/SKILL.md")).expect("sections to move");
        let kept = headings(&chat.prompt);
        let moved = headings(&chat.skill_md);
        for section in ON_DEMAND_SECTIONS {
            assert!(moved.contains(section), "{section} not moved");
            assert!(!kept.contains(section), "{section} still in prompt");
        }
        // Nothing is lost: every original section lands in exactly one place.
        for section in headings(BASE) {
            assert_eq!(
                usize::from(kept.contains(&section)) + usize::from(moved.contains(&section)),
                1,
                "{section}"
            );
        }
        for core in [
            "Incoming Turn Contract",
            "Communication Patterns",
            "Skill Safety",
            "Agent Memory",
        ] {
            assert!(kept.contains(&core), "{core} must stay in the prompt");
        }
        assert!(chat.prompt.contains("/nest/SKILL.md"));
        assert!(chat.skill_md.starts_with("---\nname: buzz-workflows\n"));
    }

    #[test]
    fn chat_profile_is_materially_smaller() {
        let chat = chat_base(BASE, Path::new("/nest/SKILL.md")).unwrap();
        assert!(
            chat.prompt.len() * 10 < BASE.len() * 7,
            "chat {} vs full {} bytes",
            chat.prompt.len(),
            BASE.len()
        );
    }

    #[test]
    fn chat_profile_installs_the_skill_and_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let prompt = apply(PromptProfile::Chat, BASE, dir.path());
        let path = skill_path(dir.path());
        let written = std::fs::read_to_string(&path).unwrap();
        assert!(prompt.contains(&path.display().to_string()));
        assert!(written.contains("## Engineering Discipline"));
        assert_eq!(apply(PromptProfile::Chat, BASE, dir.path()), prompt);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), written);
    }

    #[test]
    fn chat_profile_falls_back_to_full_when_the_skill_cannot_be_written() {
        let dir = tempfile::tempdir().unwrap();
        // A file where the skills directory should be makes the write fail.
        std::fs::write(dir.path().join(".agents"), "not a directory").unwrap();
        assert_eq!(apply(PromptProfile::Chat, BASE, dir.path()), BASE);
    }

    #[test]
    fn custom_base_without_on_demand_sections_is_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let custom = "You are a greeter.\n\n## Tone\n\nBe kind.\n";
        assert_eq!(apply(PromptProfile::Chat, custom, dir.path()), custom);
        assert!(!skill_path(dir.path()).exists());
    }
}
