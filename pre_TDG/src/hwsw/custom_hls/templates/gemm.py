"""gemm: C[M, N] = A[M, K] . B[K, N]. B is held on chip whole (no tiling in this model)."""

from hwsw.custom_hls.templates.batching import batch_loops, batch_shape


def expand(params: dict) -> dict:
    m, n, k = params["M"], params["N"], params["K"]
    batch, dtype = params.get("batch", 1), params["dtype"]
    return {
        "arrays": {
            "a": {"shape": [*batch_shape(batch), m, k], "dtype": dtype, "dir": "in"},
            "b": {"shape": [k, n], "dtype": dtype, "dir": "in"},
            "c": {"shape": [*batch_shape(batch), m, n], "dtype": dtype, "dir": "out"},
        },
        "loops": [
            *batch_loops(batch),
            {"var": "m", "range": m},
            {"var": "n", "range": n},
            {"var": "k", "range": k},
        ],
        "body": {"mac": 1},
        "access": {
            "a": {"pattern": "streaming"},
            "b": {"pattern": "reuse"},
            "c": {"pattern": "streaming"},
        },
        "hints": {"unrollable": ["k"]},
    }
