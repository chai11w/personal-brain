# Security policy

## Do not submit sensitive data

Do not open issues, pull requests, or test fixtures containing real memories, reports, database files, Router exports, credentials, account identifiers, private URLs, or screenshots from a real runtime.

Use synthetic data created from scratch. Renaming or paraphrasing a real memory is not sufficient anonymization.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting feature when available. Do not post secrets or personal data in a public issue.

## Local secret handling

Configuration files contain environment-variable names only. Store values in the operating-system environment or a dedicated secret manager. The offline demo and CI require no credentials.

## Publication checks

Before publishing a commit or release:

1. Run `python scripts/security/check_public_paths.py`.
2. Run the test suite.
3. Scan the full Git history and worktree with Gitleaks.
4. Review demo text and screenshots manually for semantic privacy leaks.
