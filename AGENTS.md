# AGENTS.md

## Operating rules

1. main is always shippable
- Work directly on main
- If something is wrong, move forward and fix it

2. Validation over reading
- Tests and runnable commands are the quality gate
- Prefer fast executable targets over long review cycles

3. Short prompts, tight steering
- Use a contract prompt with a small executable outcome
- Minimize refactors unless required for correctness

4. Local CI mindset
- Run `make smoke` locally and commit when green

5. Parallelize sessions
- Builder: implement the thin slice
- Verifier: tests, edge cases, smoke runs
- Janitor: docs, ergonomics, cleanup

## The loop

1) Pick the smallest executable target
- a CLI command with deterministic output
- a report artifact with fixed schema
- one dashboard element driven by a table

2) Use a contract prompt
Template:
- Goal: one sentence
- Constraints: 3 to 6 bullets
- Acceptance tests: exact commands to run
- Fence: what not to touch

3) Require proof
Agent response must include:
- exact commands it ran
- short output summary
- files changed list and why
- smallest diff that satisfies the goal

4) Run the gate locally
- default gate: `make smoke`

5) Commit immediately when correct
- small commits
- keep main green

## Commands

Makefile targets:
- `make setup`
- `make update`
- `make features`
- `make train`
- `make signals`
- `make eval`
- `make report`
- `make export-alerts`
- `make dashboard`
- `make test`
- `make daily`
- `make smoke`

Console scripts (installed into the venv):
- `oil-update-market`
- `oil-update-news`
- `oil-build-features`
- `oil-train-model`
- `oil-generate-signals`
- `oil-evaluate-signals`
- `oil-generate-report`
- `oil-export-alerts`
- `oil-dashboard`

## Repo fences

- Do not change schemas lightly. If you change a table, add a migration path or a reset-db note in docs, plus tests.
- Do not add dependencies unless required for correctness and the benefit is clear.
- Tests must not call the network unless explicitly mocked.
- Keep artifacts and outputs stable:
  - reports: `reports/`
  - alerts: `artifacts/alerts.json`
  - DB: `data/oil_risk.db`
