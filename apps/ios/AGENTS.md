# iOS scope

`CLAUDE.md` is the canonical Claude Code context for this scope. Keep the
sibling `AGENTS.md` byte-identical as a compatibility mirror. Read
`../CLAUDE.md` and the repository-root `CLAUDE.md` first.

- The app is SwiftUI and local-first. Treat current Swift source, the Xcode
  project, and `docs/apps/00-status.md` as authoritative; do not infer the live
  pipeline from old architecture prose.
- Raw practice audio is uploaded to the backend and authoritative scoring
  returns over the practice WebSocket. Do not assume the inert on-device Core ML
  path is the source of truth.
- Use Serena/sourcekit-lsp for symbol definitions and references. Fall back to
  text search only when the language server cannot resolve the file.
- New Swift files must be registered in
  `CrescendAI.xcodeproj/project.pbxproj`. Run `just check-ios` to catch files on
  disk that Xcode would silently omit.
- Follow existing `@Observable`, actor, async/await, and `MainActor` patterns.
  Surface errors explicitly and capture them at call sites with the existing
  Sentry integration.
- Changes to capture, synchronization, API contracts, or the user-visible
  practice flow require a human-lit simulator/device click-through.

Route product behavior to `docs/apps/02-pipeline.md`, UI decisions to
`docs/apps/05-ui-system.md`, and current capability/evaluation claims to
`docs/apps/06-capabilities.md` and `docs/apps/07-evaluation.md`.
