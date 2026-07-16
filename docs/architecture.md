# Architecture and engineering decisions

## Data ownership

SQLite is the source of truth. Markdown reports and Router JSON files are derived views and must be rebuildable. Real derived views are private because summaries and topics can reveal as much as raw notes.

## Write path

Raw input is persisted before model extraction. Each extraction run records model identity, prompt version, input hash, output JSON, status, and error information. Atomic memories retain references to both raw input and extraction run.

## Read path

The current retrieval path embeds the query, scores active memory vectors, applies small lexical/task/lifecycle adjustments, reranks candidates with a chat model, and produces an answer constrained to selected evidence.

This repository deliberately does not present Agent RAG as completed work. Query planning and bounded retrieval reflection should only be adopted after a gold-set evaluation shows repeated routing failures that a simpler filter or hybrid search cannot solve.

## Security boundary

Application code and synthetic fixtures may be public. Real runtime data, derived memory views, reports, logs, credentials, backups, and operational agent context are private. A clean public history is built from an allowlist rather than by deleting sensitive files from an old history.

## Recovery

SQLite backups use the backup API rather than copying a live database file. Restore verification occurs in a new isolated path and checks hash, integrity, foreign keys, and table counts without overwriting production.
