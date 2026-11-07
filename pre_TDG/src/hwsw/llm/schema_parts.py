"""Pieces of the TDG contract schema, small enough to put in a prompt."""

from typing import Any

from hwsw.validation.json_schema_rules import load_tdg_schema


def with_tdg_defs(schema: dict) -> dict:
    """`schema` plus the TDG `$defs` it refers to (directly or not), and no others."""
    all_defs = load_tdg_schema()["$defs"]
    needed: dict[str, dict] = {}
    pending = list(_referenced_defs(schema))
    while pending:
        name = pending.pop()
        if name not in needed:
            needed[name] = all_defs[name]
            pending.extend(_referenced_defs(all_defs[name]))
    return {**schema, "$defs": dict(sorted(needed.items()))}


def tdg_ref(def_name: str) -> dict:
    return {"$ref": f"#/$defs/{def_name}"}


def _referenced_defs(value: Any) -> set[str]:
    if isinstance(value, dict):
        found = {value["$ref"].rsplit("/", 1)[1]} if "$ref" in value else set()
        return found.union(*map(_referenced_defs, value.values()))
    if isinstance(value, list):
        return set().union(*map(_referenced_defs, value))
    return set()
