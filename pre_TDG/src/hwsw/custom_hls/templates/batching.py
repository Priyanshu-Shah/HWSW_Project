"""Optional leading batch dimension shared by the templates (batch 1 adds nothing)."""


def batch_loops(batch: int) -> list[dict]:
    return [{"var": "n", "range": batch}] if batch > 1 else []


def batch_shape(batch: int) -> list[int]:
    return [batch] if batch > 1 else []
