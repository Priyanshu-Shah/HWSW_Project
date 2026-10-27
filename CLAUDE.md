# HW/SW Partitioning Tool

English task → TDG → costed TDG → partitioner. See `README.md`.

## Rules

- Folders by team: `pre_TDG/` (**our work**: LLM pipeline + custom_hls → costed TDG), `post_TDG/` (partitioner team; empty, not ours: create nothing there), `library/` (library team; README of expectations only), `TDG/` (shared interface).
- Work inside one folder; `pre_TDG/` has its own `CLAUDE.md`, `decisions.md`, `implementation.md`. Don't read other folders unless needed.
- **Any change to `TDG/` or to the board-file format in `library/README.md`: stop and tell the user first, so they can inform the other teams.**
- All exchange files are JSON.
- `decisions.md` holds active decisions only; keep every `CLAUDE.md` ≤150 words.
