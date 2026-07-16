# Public/private security boundary

This repository is the approved public portfolio layer. Real runtime data remains in a separate private environment and must never be copied here.

## Public allowlist

- Reviewed application source under `personal_brain/` and reviewed generic CLI/adapters.
- Synthetic demo fixtures created from scratch; never renamed or paraphrased real memories.
- Isolated tests that use temporary SQLite databases and no network or production config.
- Public architecture, security, contribution, and demo documentation.
- Placeholder configuration such as `.env.example` or `config.example.json` with no usable credentials.
- Reusable Skill definitions only after an independent content review and file-exact inclusion in the approved public allowlist. Nothing under this runtime repository's `.agents/` is path-allowlisted.

An allowlisted location is not automatically safe: each file still needs content and secret review before publication.

## Private denylist

- Real databases, journals, backups, dumps, logs, reports, caches, and temporary test output.
- `brain_index.json`, `memory/*.json`, embeddings, Router exports, and any other derived real-memory view.
- Private `.agents` project context, audit reports, handoff files, backlog, stabilization logs, and local security reports.
- `.env`, `config.json`, local configuration, tokens, credentials, tunnel/account identifiers, and encrypted credential blobs.
- Screenshots, transcripts, fixtures, or benchmarks produced from real runtime data.
- Git bundles, working-tree snapshots, recovery manifests, and restore-test artifacts.

## Publication rule

Maintain this public portfolio from the reviewed allowlist only. Before every release or push, run the path guard, tests, and a dedicated full-history secret scanner. Any change that expands the allowed data or file scope requires an independent privacy review and explicit authorization.

The local path guard is intentionally conservative and is not a substitute for a dedicated secret scanner or semantic privacy review.

Backup, restore, and worktree-capture tools reject output inside a Git worktree unless the exact target is ignored. Ignored files are inventoried by worktree capture but are not copied implicitly: databases use the SQLite backup tool, while credentials and other runtime material require a separately authorized private backup strategy.
