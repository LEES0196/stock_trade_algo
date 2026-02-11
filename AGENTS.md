# Repository Guidelines

## Project Structure & Module Organization
- `src/` — application code organized by feature (e.g., `src/orders/`, `src/portfolio/`).
- `tests/` — mirrors `src/` with test files (e.g., `tests/orders/test_api.py`).
- `scripts/` — developer utilities (data loaders, one-offs).
- `docs/` — architecture notes, ADRs, and usage guides.
- `assets/` or `data/` — static files and sample datasets (checked in only if small).

## Build, Test, and Development Commands
- `make install` — install dependencies for the active stack (Python/JS). 
- `make run` — start the local app or CLI entrypoint.
- `make test` — run the full test suite with coverage.
- `make lint` — run linters (style + static analysis).
- `make fmt` — auto-format the codebase.

Tip: If no Makefile exists yet, run the underlying tools directly (e.g., `pytest -q`, `ruff check .`, `black .`, or `npm test`).

## Coding Style & Naming Conventions
- Prefer clear, descriptive names: `snake_case` for Python, `camelCase` for JS, `PascalCase` for classes.
- Keep functions focused; aim for <50 lines where practical.
- Python: format with `black` (line length 88) and lint with `ruff`.
- JS/TS: format with `prettier` and lint with `eslint`.
- Organize modules by feature domain rather than layer when possible.

## Testing Guidelines
- Framework: `pytest` for Python; `jest`/`vitest` for JS/TS.
- Place tests under `tests/` mirroring `src/` paths.
- Name tests `test_*.py` (Python) or `*.test.ts`/`*.test.js`.
- Target >90% coverage for core modules; add regression tests for bugs.
- Run `make test` before any push.

## Commit & Pull Request Guidelines
- Use Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
- Keep commits small and scoped; include rationale in the body when non-trivial.
- PRs must include: purpose, linked issues, test coverage notes, and screenshots/logs for UX/CLI changes.
- CI must pass (lint, format, tests) before merge.

## Security & Configuration
- Do not commit secrets; use environment variables and `.env.example` for placeholders.
- Validate inputs at boundaries; treat external data as untrusted.
- Document any required API keys or data sources in `docs/config.md`.

## Agent-Specific Instructions
- This file governs the entire repo. When adding files, follow the structure above and keep changes minimal, focused, and well-tested.
