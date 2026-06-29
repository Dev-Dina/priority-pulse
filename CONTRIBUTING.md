# Contributing

## Commit convention — Conventional Commits

Format: `type(scope): imperative lowercase summary`

- No trailing period.
- No body unless it is genuinely needed to explain *why*.
- Never add an AI trailer, `Co-Authored-By`, or emoji footer.

| Type | Use for |
|----------|---------|
| feat | a new capability or user-visible behaviour |
| fix | a bug fix or behavioural correction |
| perf | a change that only improves performance |
| refactor | code change that neither fixes a bug nor adds a feature |
| docs | documentation, docstring, or comment-only changes |
| test | adding or changing tests |
| build | build system, dependencies, Docker/image, packaging |
| ci | CI configuration and pipelines |
| chore | maintenance that doesn't change source behaviour (config, ignores) |

`scope` is optional but encouraged — e.g. `llm`, `rag`, `api`, `ml`, `deps`, `docker`.

Examples:
- `feat(llm): add groq fallback`
- `fix(rag): embed complaint-only text to stay in complaint space`
- `test(ml): cover engineer_features and label_ticket`

## Code standards (held going forward)

- **Lint + format:** [ruff](https://docs.astral.sh/ruff/). Run `ruff check` and
  `ruff format` before committing; CI enforces both.
- **Type hints** on every public function — parameters and return type.
- **Docstrings:** concise. State what a function does and any non-obvious
  contract; skip restating the obvious.
- **No committed dead code** and **no scaffolding `TODO`/placeholder comments**.
  Track follow-ups in the issue tracker, not in source.
- Keep configuration in `config.py` (pydantic-settings), not hardcoded in logic.
