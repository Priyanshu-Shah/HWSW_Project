"""Write JSON Schema files for the contract models.

Usage: python -m schemas.export   (writes schemas/json/*.schema.json)
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from schemas.ir import TaskGraph
from schemas.platform import Platform
from schemas.result import Ranking, Result

MODELS: dict[str, type[BaseModel]] = {
    "ir": TaskGraph,
    "platform": Platform,
    "result": Result,
    "ranking": Ranking,
}
OUT_DIR = Path(__file__).resolve().parent / "json"


def render(name: str) -> str:
    schema = MODELS[name].model_json_schema()
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": f"{name}.schema.json", **schema}
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for name in MODELS:
        (OUT_DIR / f"{name}.schema.json").write_text(render(name), encoding="utf-8")
        print(f"wrote {OUT_DIR / f'{name}.schema.json'}")


if __name__ == "__main__":
    main()
