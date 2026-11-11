"""One folder per pipeline run, with a JSONL log of every LLM call and tool call."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from hwsw.repo_paths import RUNS_DIR

LOG_FILE_NAME = "llm.jsonl"


def new_run_dir(name: str, runs_dir: Path = RUNS_DIR) -> Path:
    """`runs/<timestamp>_<name>/`, created empty."""
    run_dir = runs_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


class RunLog:
    """Appends one JSON line per record. With no path, records are dropped (tests, quick calls)."""

    def __init__(self, path: Path | None = None):
        self.path = path

    def write(self, kind: str, **fields: Any) -> None:
        if self.path is None:
            return
        record = {"time": time.time(), "kind": kind, **fields}
        with self.path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
