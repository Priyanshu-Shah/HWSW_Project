# TDG/

Shared interface: produced by `pre_TDG/`, consumed by `post_TDG/`; board specs come from `library/`.

- **Before changing anything here (schemas, examples, D1/D3/D4/D5): stop and tell the user**, so they can inform the other teams. Bump `schema_version`; keep examples valid.
- Validate edits: every file in `schemas/examples/` must pass `schemas/tdg.schema.json`.
- `decisions.md` = active shared decisions only. `implementation.md` = contract status and open items.
- Don't add team-specific logic here.
