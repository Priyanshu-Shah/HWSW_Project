"""memcpy: copy `bytes` bytes, a 4-byte word per iteration. No arithmetic, only data movement."""

import math

WORD_BYTES = 4


def expand(params: dict) -> dict:
    words = math.ceil(params["bytes"] / WORD_BYTES)
    return {
        "arrays": {
            "source": {"shape": [words], "dtype": "int32", "dir": "in"},
            "destination": {"shape": [words], "dtype": "int32", "dir": "out"},
        },
        "loops": [{"var": "i", "range": words}],
        "body": {"load": 1, "store": 1},
        "access": {"source": {"pattern": "streaming"}, "destination": {"pattern": "streaming"}},
    }
