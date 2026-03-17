# Contributing to orya-guard

Thanks for contributing to `orya-guard`.

This project is maintained in the **OryaOne** organization and published by **Dimitris Kampouridis**. The goal is a small, reliable open-source tool for preflight validation of ML datasets and inference payloads.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run checks locally

```bash
ruff format .
ruff check .
pytest
```

If you prefer shortcuts, the `Makefile` includes:

```bash
make install
make format
make lint
make test
make check
```

## Development guidelines

- Keep the project focused on preflight validation for datasets and inference payloads.
- Prefer small, readable changes over broad refactors.
- Add or update tests when behavior changes.
- Keep CLI text and docs concise, consistent, and beginner-friendly.

## Proposing changes

For small fixes, open a pull request directly.

For larger changes:

1. Open an issue first.
2. Explain the problem and the proposed approach.
3. Keep scope tight and aligned with the project vision.

## Pull request checklist

Before opening a PR, make sure:

- `ruff format .` has been run
- `ruff check .` passes
- `pytest` passes
- documentation is updated if the CLI or behavior changed
- the change is focused and easy to review

## Questions

If you are unsure whether a feature fits the project, open an issue before implementing it.
