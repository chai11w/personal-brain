# Synthetic demo data policy

All demo records must be created from scratch for this public repository.

Allowed:

- fictional people, organizations, projects, dates, tasks, and decisions;
- deterministic synthetic IDs and vectors;
- examples that demonstrate product behavior without resembling a real user's history.

Not allowed:

- real records with names changed;
- paraphrased or translated real memories;
- exported reports, Router files, logs, screenshots, or embeddings from a real runtime;
- production identifiers, paths, URLs, account names, or timestamps.

Every fixture must include `"synthetic": true` and a short provenance note.
