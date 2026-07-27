# Claude Code entry point

Read and follow `AGENTS.md`; it is the canonical shared CrescendAI context.

Claude-specific delta:

- The primary-tree edit guard in `.claude/hooks/guard-primary-tree-edits.py`
  blocks edits on `main`/`master`. If it fires, enter the issue worktree.
- When a scoped `CLAUDE.md` points to a scoped `AGENTS.md`, read that
  `AGENTS.md` before editing in the directory.
- Use the named Claude skills for the workflow routed from `AGENTS.md`; do not
  duplicate those procedures here.
