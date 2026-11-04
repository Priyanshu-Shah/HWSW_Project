"""Compact id lists for error messages ("existing node ids: n1, n2, ...")."""

from collections.abc import Iterable


def describe_ids(ids: Iterable[str], limit: int = 10) -> str:
    ids = list(ids)
    if len(ids) <= limit:
        return ", ".join(ids)
    return f"{', '.join(ids[:limit])}, ... ({len(ids)} in total)"
