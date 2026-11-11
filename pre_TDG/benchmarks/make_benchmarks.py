"""Writes the benchmark tasks (tasks/*.json) and their reference TDGs (reference/*.tdg.json).

The references are hand-made: one plausible, reasonably sized graph per task, not the only right
answer. They are defined here rather than typed as JSON so that every byte count and op count
is computed, not written by hand. Edit this file and run it again:

    .venv/bin/python benchmarks/make_benchmarks.py

Each node has `roles`: words that a generated node's name may contain when it does the same job.
Scoring uses them to match generated nodes to reference nodes.
"""

import math
from pathlib import Path

from hwsw.json_files import write_json

HERE = Path(__file__).parent
DTYPE_BYTES = {"int8": 1, "uint8": 1, "int16": 2, "int32": 4, "fp16": 2, "fp32": 4}
NOT_OPS = ("load", "store")


def opaque(node_id, name, sw_time_s, roles, phase="iteration"):
    node = {"id": node_id, "name": name, "phase": phase, "affinity": "sw_only", "kind": "opaque"}
    node["opaque"] = {"sw_time_s": sw_time_s, "source": "library_default"}
    return node, roles


def template(node_id, name, op, params, roles, phase="iteration"):
    node = {"id": node_id, "name": name, "phase": phase, "affinity": "any", "kind": "template"}
    node["template"] = {"op": op, "params": params}
    return node, roles


def loop_nest(node_id, name, roles, *, arrays, loops, body, phase="iteration", **extra):
    """`loops` is [(var, range), ...]; `extra` holds access, control, hints."""
    nest = {
        "arrays": arrays,
        "loops": [{"var": var, "range": size} for var, size in loops],
        "body": body,
        **extra,
    }
    iterations = math.prod(size for _, size in loops)
    ops_unit = "macs" if set(body) == {"mac"} else "ops"
    per_iteration = sum(count for op, count in body.items() if op not in NOT_OPS)
    node = {"id": node_id, "name": name, "phase": phase, "affinity": "any", "kind": "loop_nest"}
    node.update(loop_nest=nest, ops_claimed=iterations * per_iteration, ops_unit=ops_unit)
    return node, roles


def array(shape, dtype, direction, const=False):
    declared = {"shape": list(shape), "dtype": dtype, "dir": direction}
    return {**declared, "const": True} if const else declared


def size(shape, dtype):
    return math.prod(shape) * DTYPE_BYTES[dtype]


def benchmark(
    task_id, title, task, *, platform, arrival, hardness, deadline_s, answers, nodes, edges
):
    """`answers`: [(keywords, answer)]; `edges`: [(src, dst, bytes)], bytes 0 = control edge."""
    graph_nodes = [node for node, _ in nodes]
    tdg = {
        "graph": {
            "id": task_id,
            "schema_version": "0.2",
            "provenance": {"kind": "manual", "source": f"benchmark reference: {title}"},
        },
        "arrival": arrival,
        "objective": {"latency": 1.0},
        "platform": platform,
        "nodes": graph_nodes,
        "edges": [
            {
                "id": f"e{index}",
                "src": src,
                "dst": dst,
                "kind": "data" if size_bytes else "control",
                "bytes": size_bytes,
            }
            for index, (src, dst, size_bytes) in enumerate(edges, start=1)
        ],
    }
    if deadline_s is not None:
        tdg["constraints"] = {"deadline_s": deadline_s}
    task_file = {
        "id": task_id,
        "title": title,
        "task": task,
        "answers": [{"keywords": keywords, "answer": answer} for keywords, answer in answers],
        "expected": {
            "platform": platform,
            "arrival_kind": arrival["kind"],
            "hardness": hardness,
            "deadline_s": deadline_s,
        },
        "roles": {node["id"]: roles for node, roles in nodes},
    }
    return task_file, tdg


def chain(*node_ids_and_bytes):
    """chain("n1", 10, "n2", 20, "n3") -> [("n1", "n2", 10), ("n2", "n3", 20)]."""
    items = list(node_ids_and_bytes)
    return [(items[i], items[i + 2], items[i + 1]) for i in range(0, len(items) - 2, 2)]


# Question words about timing. Not "hard" alone: it would also match "hardware".
TIMING_KEYWORDS = [
    "hard or soft",
    "soft",
    "deadline",
    "strict",
    "every frame",
    "real-time",
    "real time",
    "miss",
    "drop",
    "timing",
    "hardness",
]

# Answers most tasks share; task-specific answers come first and win.
BOARD_ANSWERS = {
    board: [(["board", "platform", "fpga", "device"], board)]
    for board in ("pynq_z2", "kv260", "zcu104")
}


def t01_object_detection():
    frame, model_in = (720, 1280, 3), (640, 640, 3)

    def conv(node_id, name, roles, h, c_in, c_out, k, stride):
        params = {"H": h, "W": h, "C_in": c_in, "C_out": c_out, "K": k, "stride": stride}
        return template(
            node_id, name, "conv2d", {**params, "padding": k // 2, "dtype": "int8"}, roles
        )

    return benchmark(
        "t01_object_detection",
        "Object detection, 720p @ 30 fps",
        "Real-time object detection on a 720p camera stream at 30 fps on a ZCU104.",
        platform="zcu104",
        arrival={"kind": "streaming", "period_s": 1 / 30},
        hardness="hard",
        deadline_s=1 / 30,
        answers=[
            (["model input", "input size", "network input", "input resolution"], "640x640"),
            (["data type", "precision", "dtype", "quantiz", "bit"], "int8"),
            (["class"], "80 (COCO)"),
            (TIMING_KEYWORDS, "hard"),
            (["model", "network", "detector", "architecture"], "YOLOv8n"),
            (["resolution", "frame size", "camera"], "1280x720"),
            *BOARD_ANSWERS["zcu104"],
        ],
        nodes=[
            opaque("n1", "capture_frame", 0.002, ["capture", "camera", "acquire", "grab", "read"]),
            loop_nest(
                "n2",
                "resize_letterbox",
                ["resize", "letterbox", "scale", "preprocess"],
                arrays={
                    "frame": array(frame, "uint8", "in"),
                    "img": array(model_in, "uint8", "out"),
                },
                loops=[("y", 640), ("x", 640), ("c", 3)],
                body={"mac": 4},
                access={"frame": {"pattern": "sliding_window", "reuse": ["x"]}},
                hints={"unrollable": ["c"]},
            ),
            template(
                "n3",
                "normalize",
                "elementwise",
                {
                    "elements": math.prod(model_in),
                    "ops_per_element": 1,
                    "inputs": 1,
                    "dtype": "int8",
                },
                ["normalize", "normalise", "quantize", "scale", "preprocess"],
            ),  # fmt: skip
            template(
                "n4",
                "load_weights",
                "memcpy",
                {"bytes": 3_200_000},
                ["weight", "load", "model"],
                phase="setup",
            ),  # fmt: skip
            conv("n5", "backbone_stem", ["stem", "backbone", "conv", "stage"], 640, 3, 32, 3, 2),
            conv("n6", "backbone_p3", ["backbone", "stage", "conv", "block"], 320, 32, 64, 3, 2),
            conv("n7", "backbone_p4", ["backbone", "stage", "conv", "block"], 160, 64, 128, 3, 2),
            conv("n8", "backbone_p5", ["backbone", "stage", "conv", "block"], 80, 128, 256, 3, 2),
            conv("n9", "neck", ["neck", "fpn", "pan", "feature", "conv"], 40, 256, 256, 3, 1),
            conv("n10", "detect_head", ["head", "detect", "predict", "conv"], 40, 256, 144, 1, 1),
            loop_nest(
                "n11",
                "decode_boxes",
                ["decode", "box", "score", "threshold", "postprocess", "post"],
                arrays={
                    "head_out": array((40, 40, 144), "int8", "in"),
                    "candidates": array((300, 6), "fp32", "out"),
                },
                loops=[("anchor", 1600), ("cls", 80)],
                body={"cmp": 1},
                access={"head_out": {"pattern": "streaming"}},
                control={"branches": 1},
            ),
            loop_nest(
                "n12",
                "non_max_suppression",
                ["nms", "suppression", "non", "postprocess", "filter"],
                arrays={
                    "candidates": array((300, 6), "fp32", "in"),
                    "detections": array((100, 6), "fp32", "out"),
                },
                loops=[("i", 300), ("j", 300)],
                body={"mul": 4, "add": 6, "div": 1, "cmp": 2},
                access={"candidates": {"pattern": "reuse", "reuse": ["j"]}},
                control={"branches": 1},
            ),
            opaque(
                "n13",
                "publish_detections",
                0.0005,
                ["output", "publish", "draw", "display", "send", "write"],
            ),
        ],
        edges=[
            *chain(
                "n1",
                size(frame, "uint8"),
                "n2",
                size(model_in, "uint8"),
                "n3",
                size(model_in, "int8"),
                "n5",
                320 * 320 * 32,
                "n6",
                160 * 160 * 64,
                "n7",
                80 * 80 * 128,
                "n8",
                40 * 40 * 256,
                "n9",
                40 * 40 * 256,
                "n10",
                40 * 40 * 144,
                "n11",
                size((300, 6), "fp32"),
                "n12",
                size((100, 6), "fp32"),
                "n13",
            ),  # fmt: skip
            ("n4", "n5", 0),
        ],
    )


def t02_sobel():
    rgb, gray, edges_shape = (1080, 1920, 3), (1080, 1920), (1078, 1918)
    return benchmark(
        "t02_sobel_edges",
        "Sobel edge detection, 1080p @ 60 fps",
        "Sobel edge detection on a 1080p video stream at 60 fps on the Kria KV260.",
        platform="kv260",
        arrival={"kind": "streaming", "period_s": 1 / 60},
        hardness="hard",
        deadline_s=1 / 60,
        answers=[
            (["data type", "precision", "dtype", "bit", "pixel format"], "uint8"),
            (["threshold", "binar"], "yes, a fixed threshold on the gradient magnitude"),
            (TIMING_KEYWORDS, "hard"),
            (["color", "colour", "gray", "channel"], "RGB input, converted to grayscale"),
            *BOARD_ANSWERS["kv260"],
        ],
        nodes=[
            opaque(
                "n1",
                "capture_frame",
                0.001,
                ["capture", "camera", "acquire", "grab", "read", "input"],
            ),
            loop_nest(
                "n2",
                "to_grayscale",
                ["gray", "grey", "luma", "color", "colour", "convert"],
                arrays={"rgb": array(rgb, "uint8", "in"), "gray": array(gray, "uint8", "out")},
                loops=[("y", 1080), ("x", 1920), ("c", 3)],
                body={"mac": 1},
                access={"rgb": {"pattern": "streaming"}, "gray": {"pattern": "streaming"}},
                hints={"unrollable": ["c"]},
            ),
            loop_nest(
                "n3",
                "sobel_gradient",
                ["sobel", "gradient", "edge", "convolution", "filter", "magnitude"],
                arrays={
                    "gray": array(gray, "uint8", "in"),
                    "magnitude": array(edges_shape, "uint8", "out"),
                },
                loops=[("y", 1078), ("x", 1918), ("ky", 3), ("kx", 3)],
                body={"mac": 2},
                access={
                    "gray": {"pattern": "sliding_window", "reuse": ["ky", "kx"]},
                    "magnitude": {"pattern": "streaming"},
                },
                hints={"unrollable": ["ky", "kx"]},
            ),
            template(
                "n4",
                "threshold",
                "elementwise",
                {
                    "elements": math.prod(edges_shape),
                    "ops_per_element": 1,
                    "inputs": 1,
                    "dtype": "uint8",
                },
                ["threshold", "binar", "mask"],
            ),  # fmt: skip
            opaque("n5", "write_output", 0.001, ["output", "write", "display", "store", "send"]),
        ],
        edges=chain(
            "n1",
            size(rgb, "uint8"),
            "n2",
            size(gray, "uint8"),
            "n3",
            size(edges_shape, "uint8"),
            "n4",
            size(edges_shape, "uint8"),
            "n5",
        ),  # fmt: skip
    )


def t03_classification():
    def conv(node_id, name, roles, h, c_in, c_out, k, stride):
        params = {"H": h, "W": h, "C_in": c_in, "C_out": c_out, "K": k, "stride": stride}
        return template(
            node_id, name, "conv2d", {**params, "padding": k // 2, "dtype": "int8"}, roles
        )

    image = (480, 640, 3)
    return benchmark(
        "t03_image_classification",
        "Image classification (MobileNetV2), one-shot",
        "Classify a single 640x480 photo with MobileNetV2 on a PYNQ-Z2 board.",
        platform="pynq_z2",
        arrival={"kind": "one_shot"},
        hardness="none",
        deadline_s=None,
        answers=[
            (["input size", "model input", "network input", "input resolution"], "224x224"),
            (["data type", "precision", "dtype", "quantiz", "bit"], "int8"),
            (["class"], "1000 (ImageNet)"),
            ([*TIMING_KEYWORDS, "latency", "fast"], "none, as fast as possible"),
            (["batch", "how many", "images"], "1"),
            *BOARD_ANSWERS["pynq_z2"],
        ],
        nodes=[
            opaque("n1", "load_image", 0.005, ["load", "read", "image", "file", "decode", "input"]),
            loop_nest(
                "n2",
                "resize_normalize",
                ["resize", "normalize", "normalise", "scale", "preprocess", "crop"],
                arrays={
                    "image": array(image, "uint8", "in"),
                    "tensor": array((224, 224, 3), "int8", "out"),
                },
                loops=[("y", 224), ("x", 224), ("c", 3)],
                body={"mac": 4, "add": 1},
                access={"image": {"pattern": "sliding_window", "reuse": ["x"]}},
                hints={"unrollable": ["c"]},
            ),
            conv("n3", "stem_conv", ["stem", "conv", "first", "backbone"], 224, 3, 32, 3, 2),
            conv(
                "n4",
                "blocks_112",
                ["block", "bottleneck", "stage", "inverted", "backbone"],
                112,
                32,
                96,
                3,
                2,
            ),
            conv(
                "n5",
                "blocks_56",
                ["block", "bottleneck", "stage", "inverted", "backbone"],
                56,
                96,
                144,
                3,
                2,
            ),
            conv(
                "n6",
                "blocks_28",
                ["block", "bottleneck", "stage", "inverted", "backbone"],
                28,
                144,
                192,
                3,
                2,
            ),
            conv(
                "n7",
                "blocks_14",
                ["block", "bottleneck", "stage", "inverted", "backbone"],
                14,
                192,
                320,
                3,
                2,
            ),
            conv(
                "n8",
                "head_conv",
                ["head", "conv", "pointwise", "last", "feature"],
                7,
                320,
                1280,
                1,
                1,
            ),
            template(
                "n9",
                "global_avg_pool",
                "pool",
                {"H": 7, "W": 7, "C": 1280, "K": 7, "stride": 7, "mode": "avg", "dtype": "int8"},
                ["pool", "average", "avg", "global"],
            ),  # fmt: skip
            template(
                "n10",
                "classifier_fc",
                "gemm",
                {"M": 1, "N": 1000, "K": 1280, "dtype": "int8"},
                ["classifier", "fc", "fully", "dense", "linear", "logits"],
            ),  # fmt: skip
            template(
                "n11",
                "top1",
                "reduce",
                {"elements": 1000, "mode": "max", "dtype": "int8"},
                ["argmax", "top", "softmax", "max", "class", "predict", "label"],
            ),  # fmt: skip
            opaque(
                "n12",
                "report_label",
                0.0001,
                ["output", "report", "print", "write", "result", "label"],
            ),
        ],
        edges=chain(
            "n1",
            size(image, "uint8"),
            "n2",
            224 * 224 * 3,
            "n3",
            112 * 112 * 32,
            "n4",
            56 * 56 * 96,
            "n5",
            28 * 28 * 144,
            "n6",
            14 * 14 * 192,
            "n7",
            7 * 7 * 320,
            "n8",
            7 * 7 * 1280,
            "n9",
            1280,
            "n10",
            1000,
            "n11",
            1,
            "n12",
        ),  # fmt: skip
    )


def t04_fir():
    block, taps = 480, 256
    return benchmark(
        "t04_fir_audio",
        "FIR filter on audio, 48 kHz",
        "Apply a 256-tap FIR low-pass filter to a mono 48 kHz audio stream on a PYNQ-Z2, "
        "processing it in 10 ms blocks.",
        platform="pynq_z2",
        arrival={"kind": "streaming", "period_s": 0.01},
        hardness="hard",
        deadline_s=0.01,
        answers=[
            (["data type", "precision", "dtype", "bit", "sample format"], "int16"),
            (["block", "buffer", "frame size", "samples per"], "480 samples (10 ms)"),
            (TIMING_KEYWORDS, "hard"),
            (["channel", "mono", "stereo"], "1 (mono)"),
            *BOARD_ANSWERS["pynq_z2"],
        ],
        nodes=[
            opaque(
                "n1", "read_audio_block", 0.0002, ["read", "capture", "input", "acquire", "receive"]
            ),
            loop_nest(
                "n2",
                "fir_filter",
                ["fir", "filter", "convolution", "lowpass", "low"],
                arrays={
                    "x": array((block + taps - 1,), "int16", "in"),
                    "h": array((taps,), "int16", "in", const=True),
                    "y": array((block,), "int16", "out"),
                },
                loops=[("n", block), ("k", taps)],
                body={"mac": 1},
                access={
                    "x": {"pattern": "sliding_window", "reuse": ["k"]},
                    "h": {"pattern": "reuse", "reuse": ["n"]},
                },
                hints={"unrollable": ["k"]},
            ),
            template(
                "n3",
                "gain_clip",
                "elementwise",
                {"elements": block, "ops_per_element": 2, "inputs": 1, "dtype": "int16"},
                ["gain", "clip", "saturate", "scale", "volume"],
            ),  # fmt: skip
            opaque("n4", "write_audio_block", 0.0002, ["write", "output", "play", "send"]),
        ],
        edges=chain("n1", (block + taps - 1) * 2, "n2", block * 2, "n3", block * 2, "n4"),
    )


def t05_radar():
    chirps, samples = 256, 512
    cube = (chirps, samples)
    return benchmark(
        "t05_radar_range",
        "FFT-based radar range processing",
        "FMCW radar range processing on a ZCU104: each frame has 256 chirps of 512 ADC samples, "
        "20 frames per second. Range FFT per chirp, then CFAR detection of targets.",
        platform="zcu104",
        arrival={"kind": "streaming", "period_s": 0.05},
        hardness="hard",
        deadline_s=0.05,
        answers=[
            (["data type", "precision", "dtype", "bit", "adc"], "int16 samples, int32 FFT output"),
            (["window"], "Hann window"),
            (["cfar", "detection", "guard", "training"], "CA-CFAR, 32 training cells"),
            (TIMING_KEYWORDS, "hard"),
            *BOARD_ANSWERS["zcu104"],
        ],
        nodes=[
            opaque("n1", "adc_capture", 0.001, ["adc", "capture", "acquire", "read", "input"]),
            template(
                "n2",
                "apply_window",
                "elementwise",
                {"elements": chirps * samples, "ops_per_element": 1, "inputs": 2, "dtype": "int16"},
                ["window", "hann", "hamming", "taper"],
            ),  # fmt: skip
            loop_nest(
                "n3",
                "range_fft",
                ["fft", "range", "fourier", "spectrum"],
                arrays={
                    "samples": array(cube, "int16", "in"),
                    "twiddle": array((samples // 2, 2), "int16", "in", const=True),
                    "spectrum": array((chirps, samples, 2), "int32", "out"),
                },
                loops=[("chirp", chirps), ("stage", 9), ("butterfly", samples // 2)],
                body={"mul": 4, "add": 6},
                access={
                    "samples": {"pattern": "streaming"},
                    "twiddle": {"pattern": "reuse", "reuse": ["chirp"]},
                },
                hints={"unrollable": ["butterfly"]},
            ),
            loop_nest(
                "n4",
                "magnitude",
                ["magnitude", "abs", "power", "modulus"],
                arrays={
                    "spectrum": array((chirps, samples, 2), "int32", "in"),
                    "power": array(cube, "int32", "out"),
                },
                loops=[("chirp", chirps), ("bin", samples)],
                body={"mul": 2, "add": 1},
                access={"spectrum": {"pattern": "streaming"}},
            ),
            template(
                "n5",
                "integrate_chirps",
                "reduce",
                {"elements": chirps * samples, "mode": "sum", "outputs": samples, "dtype": "int32"},
                ["integrat", "sum", "average", "accumulate", "combine"],
            ),  # fmt: skip
            loop_nest(
                "n6",
                "cfar_detect",
                ["cfar", "detect", "threshold", "peak"],
                arrays={
                    "profile": array((samples,), "int32", "in"),
                    "targets": array((64, 2), "int32", "out"),
                },
                loops=[("cell", samples), ("window", 32)],
                body={"add": 1, "cmp": 1},
                access={"profile": {"pattern": "sliding_window", "reuse": ["window"]}},
                control={"branches": 1},
            ),
            opaque(
                "n7", "report_targets", 0.0005, ["report", "output", "send", "publish", "write"]
            ),
        ],
        edges=chain(
            "n1",
            size(cube, "int16"),
            "n2",
            size(cube, "int16"),
            "n3",
            size((chirps, samples, 2), "int32"),
            "n4",
            size(cube, "int32"),
            "n5",
            samples * 4,
            "n6",
            64 * 2 * 4,
            "n7",
        ),  # fmt: skip
    )


def t06_aes():
    chunk = 1 << 20  # bytes per iteration
    blocks = chunk // 16
    return benchmark(
        "t06_aes_stream",
        "AES-128 encryption of a 1 Gb/s stream",
        "Encrypt a 1 Gb/s network stream with AES-128 in CTR mode on a KV260, in 1 MiB chunks.",
        platform="kv260",
        arrival={"kind": "streaming", "period_s": chunk * 8 / 1e9},
        hardness="hard",
        deadline_s=chunk * 8 / 1e9,
        answers=[
            (["chunk", "packet", "block size", "buffer"], "1 MiB"),
            (["mode", "ctr", "cbc", "gcm"], "CTR"),
            (["key"], "one 128-bit key per session, set once"),
            ([*TIMING_KEYWORDS, "line rate", "keep up"], "hard"),
            *BOARD_ANSWERS["kv260"],
        ],
        nodes=[
            loop_nest(
                "n1",
                "key_expansion",
                ["key", "expansion", "schedule", "round"],
                phase="setup",
                arrays={
                    "key": array((16,), "uint8", "in"),
                    "round_keys": array((11, 16), "uint8", "out"),
                },
                loops=[("word", 44)],
                body={"logic": 4, "shift": 1},
            ),
            opaque(
                "n2",
                "receive_chunk",
                0.0005,
                ["receive", "read", "input", "capture", "rx", "network"],
            ),
            loop_nest(
                "n3",
                "aes_ctr_encrypt",
                ["aes", "encrypt", "cipher", "ctr", "round"],
                arrays={
                    "plaintext": array((blocks, 16), "uint8", "in"),
                    "round_keys": array((11, 16), "uint8", "in", const=True),
                    "sbox": array((256,), "uint8", "in", const=True),
                    "ciphertext": array((blocks, 16), "uint8", "out"),
                },
                loops=[("block", blocks), ("round", 10), ("byte", 16)],
                body={"logic": 4, "shift": 1},
                access={
                    "plaintext": {"pattern": "streaming"},
                    "round_keys": {"pattern": "reuse", "reuse": ["block"]},
                    "sbox": {"pattern": "random"},
                },
                hints={"unrollable": ["byte"]},
            ),
            opaque(
                "n4", "send_chunk", 0.0005, ["send", "write", "output", "transmit", "tx", "network"]
            ),
        ],
        edges=[("n1", "n3", 176), *chain("n2", chunk, "n3", chunk, "n4")],
    )


def t07_matmul():
    n, batch = 1024, 8
    matrices = batch * n * n * 4
    return benchmark(
        "t07_matmul_batch",
        "Matrix multiply 1024x1024, batch",
        "Multiply a batch of 8 pairs of 1024x1024 fp32 matrices on a ZCU104, as fast as possible.",
        platform="zcu104",
        arrival={"kind": "one_shot"},
        hardness="none",
        deadline_s=None,
        answers=[
            (["data type", "precision", "dtype", "bit"], "fp32"),
            (["batch", "how many", "pairs"], "8"),
            ([*TIMING_KEYWORDS, "latency", "fast"], "none, as fast as possible"),
            *BOARD_ANSWERS["zcu104"],
        ],
        nodes=[
            opaque("n1", "load_matrices", 0.05, ["load", "read", "input", "file"]),
            template(
                "n2",
                "batched_gemm",
                "gemm",
                {"M": n, "N": n, "K": n, "batch": batch, "dtype": "fp32"},
                ["gemm", "matmul", "multiply", "matrix", "product"],
            ),  # fmt: skip
            opaque("n3", "store_results", 0.025, ["store", "write", "output", "save"]),
        ],
        edges=chain("n1", 2 * matrices, "n2", matrices, "n3"),
    )


def t08_jpeg():
    pixels = 1080 * 1920
    blocks = pixels // 64 + 2 * (540 * 960 // 64)
    coefficients = blocks * 64
    return benchmark(
        "t08_jpeg_encode",
        "JPEG encoding of 1080p frames",
        "JPEG-encode 1080p camera frames at 30 fps on a ZCU104 (baseline JPEG, 4:2:0).",
        platform="zcu104",
        arrival={"kind": "streaming", "period_s": 1 / 30},
        hardness="soft",
        deadline_s=1 / 30,
        answers=[
            (["quality", "quantiz"], "quality 85, standard tables"),
            (["subsampl", "chroma", "4:2"], "4:2:0"),
            (TIMING_KEYWORDS, "soft"),
            (["data type", "precision", "dtype", "bit"], "uint8 pixels, int16 coefficients"),
            *BOARD_ANSWERS["zcu104"],
        ],
        nodes=[
            opaque("n1", "capture_frame", 0.002, ["capture", "camera", "acquire", "read", "input"]),
            loop_nest(
                "n2",
                "rgb_to_ycbcr",
                ["ycbcr", "color", "colour", "convert", "yuv"],
                arrays={
                    "rgb": array((1080, 1920, 3), "uint8", "in"),
                    "ycbcr": array((1080, 1920, 3), "uint8", "out"),
                },
                loops=[("y", 1080), ("x", 1920), ("c", 3)],
                body={"mac": 3},
                access={"rgb": {"pattern": "streaming"}},
                hints={"unrollable": ["c"]},
            ),
            loop_nest(
                "n3",
                "chroma_subsample",
                ["subsampl", "chroma", "downsampl", "420"],
                arrays={
                    "chroma": array((1080, 1920, 2), "uint8", "in"),
                    "subsampled": array((540, 960, 2), "uint8", "out"),
                },
                loops=[("y", 540), ("x", 960), ("c", 2), ("k", 4)],
                body={"add": 1},
                access={"chroma": {"pattern": "sliding_window", "reuse": ["k"]}},
            ),
            loop_nest(
                "n4",
                "dct_8x8",
                ["dct", "transform", "cosine"],
                arrays={
                    "blocks": array((blocks, 8, 8), "uint8", "in"),
                    "basis": array((8, 8), "int16", "in", const=True),
                    "coefficients": array((blocks, 8, 8), "int16", "out"),
                },
                loops=[("block", blocks), ("pass", 2), ("i", 8), ("j", 8), ("k", 8)],
                body={"mac": 1},
                # each 8x8 block is read once into a small local buffer
                access={"blocks": {"pattern": "streaming"}, "basis": {"pattern": "reuse"}},
                hints={"unrollable": ["k"]},
            ),
            template(
                "n5",
                "quantize",
                "elementwise",
                {"elements": coefficients, "ops_per_element": 1, "inputs": 2, "dtype": "int16"},
                ["quantiz", "quantis", "scale"],
            ),  # fmt: skip
            loop_nest(
                "n6",
                "entropy_encode",
                ["entropy", "huffman", "zigzag", "rle", "run", "encode", "coding"],
                arrays={
                    "coefficients": array((blocks, 64), "int16", "in"),
                    "bitstream": array((pixels // 4,), "uint8", "out"),
                },
                loops=[("block", blocks), ("coefficient", 64)],
                body={"logic": 3, "shift": 2, "cmp": 2},
                access={"coefficients": {"pattern": "random"}},
                control={"branches": 2, "control_heavy": True},
            ),
            opaque("n7", "write_jpeg", 0.001, ["write", "output", "save", "file", "store"]),
        ],
        edges=[
            ("n1", "n2", pixels * 3),
            ("n2", "n3", pixels * 2),
            ("n2", "n4", pixels),
            ("n3", "n4", 540 * 960 * 2),
            ("n4", "n5", coefficients * 2),
            ("n5", "n6", coefficients * 2),
            ("n6", "n7", pixels // 4),
        ],
    )


def t09_kalman():
    states, measurements = 12, 6

    def gemm(node_id, name, roles, m, n, k, batch=1):
        params = {"M": m, "N": n, "K": k, "dtype": "fp32"}
        return template(
            node_id, name, "gemm", {**params, "batch": batch} if batch > 1 else params, roles
        )

    s, z = states, measurements
    return benchmark(
        "t09_kalman_filter",
        "Kalman filter at 1 kHz, hard 1 ms limit",
        "Run a 12-state Kalman filter fusing a 6-value sensor measurement at 1 kHz on a PYNQ-Z2; "
        "every update must finish within 1 ms.",
        platform="pynq_z2",
        arrival={"kind": "periodic", "period_s": 0.001},
        hardness="hard",
        deadline_s=0.001,
        answers=[
            (["data type", "precision", "dtype", "bit", "fixed", "float"], "fp32"),
            (["state"], "12"),
            (["measurement", "sensor", "observation"], "6"),
            (TIMING_KEYWORDS, "hard"),
            *BOARD_ANSWERS["pynq_z2"],
        ],
        nodes=[
            opaque(
                "n1", "read_sensors", 0.00005, ["read", "sensor", "measure", "input", "acquire"]
            ),
            gemm("n2", "predict_state", ["predict", "state", "propagat"], s, 1, s),
            gemm(
                "n3", "predict_covariance", ["predict", "covariance", "propagat"], s, s, s, batch=2
            ),
            gemm(
                "n4",
                "innovation_covariance",
                ["innovation", "covariance", "residual"],
                z,
                z,
                s,
                batch=2,
            ),
            loop_nest(
                "n5",
                "invert_innovation",
                ["invert", "inverse", "inversion", "solve", "cholesky", "decompos"],
                arrays={
                    "s_matrix": array((z, z), "fp32", "in"),
                    "s_inverse": array((z, z), "fp32", "out"),
                },
                loops=[("pivot", z), ("row", z), ("col", 2 * z)],
                body={"mac": 1, "div": 1},
                access={"s_matrix": {"pattern": "reuse", "reuse": ["pivot"]}},
            ),
            gemm("n6", "kalman_gain", ["gain", "kalman"], s, z, z, batch=2),
            gemm("n7", "update_state", ["update", "correct", "state", "estimate"], s, 1, z),
            gemm("n8", "update_covariance", ["update", "correct", "covariance"], s, s, z),
            opaque(
                "n9", "write_estimate", 0.00002, ["write", "output", "publish", "send", "estimate"]
            ),
        ],
        edges=[
            ("n1", "n7", z * 4),
            ("n2", "n3", s * 4),
            ("n3", "n4", s * s * 4),
            ("n4", "n5", z * z * 4),
            ("n5", "n6", z * z * 4),
            ("n3", "n6", s * s * 4),
            ("n6", "n7", s * z * 4),
            ("n2", "n7", s * 4),
            ("n6", "n8", s * z * 4),
            ("n3", "n8", s * s * 4),
            ("n7", "n9", s * 4),
        ],
    )


def t10_stabilization():
    frame, gray, blurred = (720, 1280, 3), (720, 1280), (716, 1276)
    features = 200
    return benchmark(
        "t10_video_stabilization",
        "Video stabilization, 720p @ 30 fps",
        "Stabilize a shaky 720p camera stream at 30 fps on a ZCU104: track features between "
        "frames, estimate the camera motion, smooth it and warp each frame.",
        platform="zcu104",
        arrival={"kind": "streaming", "period_s": 1 / 30},
        hardness="soft",
        deadline_s=1 / 30,
        answers=[
            (
                ["feature", "corner", "keypoint", "points"],
                "200 corners (Harris), tracked with Lucas-Kanade",
            ),
            (
                ["motion model", "transform", "affine", "homography"],
                "affine, estimated with RANSAC",
            ),
            (["smooth", "window", "trajectory"], "moving average over 30 frames"),
            (TIMING_KEYWORDS, "soft"),
            (["data type", "precision", "dtype", "bit"], "uint8 pixels"),
            *BOARD_ANSWERS["zcu104"],
        ],
        nodes=[
            opaque("n1", "capture_frame", 0.002, ["capture", "camera", "acquire", "read", "input"]),
            loop_nest(
                "n2",
                "to_grayscale",
                ["gray", "grey", "luma", "convert"],
                arrays={"rgb": array(frame, "uint8", "in"), "gray": array(gray, "uint8", "out")},
                loops=[("y", 720), ("x", 1280), ("c", 3)],
                body={"mac": 1},
                access={"rgb": {"pattern": "streaming"}},
                hints={"unrollable": ["c"]},
            ),
            loop_nest(
                "n3",
                "gaussian_blur",
                ["blur", "gaussian", "smooth", "filter", "denoise", "pyramid"],
                arrays={
                    "gray": array(gray, "uint8", "in"),
                    "blurred": array(blurred, "uint8", "out"),
                },
                loops=[("y", 716), ("x", 1276), ("ky", 5), ("kx", 5)],
                body={"mac": 1},
                access={"gray": {"pattern": "sliding_window", "reuse": ["ky", "kx"]}},
                hints={"unrollable": ["ky", "kx"]},
            ),
            loop_nest(
                "n4",
                "detect_corners",
                ["corner", "feature", "harris", "detect", "keypoint", "fast", "shi"],
                arrays={
                    "blurred": array(blurred, "uint8", "in"),
                    "corners": array((features, 2), "int32", "out"),
                },
                loops=[("y", 714), ("x", 1274), ("ky", 3), ("kx", 3)],
                body={"mac": 6, "cmp": 1},
                access={"blurred": {"pattern": "sliding_window", "reuse": ["ky", "kx"]}},
                control={"branches": 1},
            ),
            loop_nest(
                "n5",
                "track_features",
                ["track", "optical", "flow", "lucas", "kanade", "match", "lk"],
                arrays={
                    "blurred": array(blurred, "uint8", "in"),
                    "corners": array((features, 2), "int32", "in"),
                    "flow": array((features, 4), "fp32", "out"),
                },
                loops=[
                    ("feature", features),
                    ("level", 3),
                    ("iteration", 10),
                    ("wy", 21),
                    ("wx", 21),
                ],
                body={"mac": 6},
                access={"blurred": {"pattern": "random"}},
                control={"branches": 1},
            ),
            loop_nest(
                "n6",
                "estimate_motion",
                ["motion", "estimate", "ransac", "affine", "transform", "homography"],
                arrays={
                    "flow": array((features, 4), "fp32", "in"),
                    "motion": array((6,), "fp32", "out"),
                },
                loops=[("trial", 500), ("point", features)],
                body={"mul": 6, "add": 6, "cmp": 1},
                access={"flow": {"pattern": "random"}},
                control={"branches": 1, "control_heavy": True},
            ),
            loop_nest(
                "n7",
                "smooth_trajectory",
                ["smooth", "trajectory", "filter", "average", "path"],
                arrays={
                    "motion": array((6,), "fp32", "in"),
                    "correction": array((6,), "fp32", "out"),
                },
                loops=[("frame", 30), ("param", 6)],
                body={"mac": 1},
            ),
            loop_nest(
                "n8",
                "warp_frame",
                ["warp", "transform", "affine", "remap", "stabiliz", "apply"],
                arrays={
                    "frame": array(frame, "uint8", "in"),
                    "correction": array((6,), "fp32", "in", const=True),
                    "stable": array(frame, "uint8", "out"),
                },
                loops=[("y", 720), ("x", 1280), ("c", 3)],
                body={"mac": 4, "mul": 4, "add": 2},
                access={"frame": {"pattern": "random"}},
            ),
            opaque("n9", "output_frame", 0.002, ["output", "write", "display", "encode", "send"]),
        ],
        edges=[
            ("n1", "n2", size(frame, "uint8")),
            ("n2", "n3", size(gray, "uint8")),
            ("n3", "n4", size(blurred, "uint8")),
            ("n3", "n5", size(blurred, "uint8")),
            ("n4", "n5", features * 2 * 4),
            ("n5", "n6", features * 4 * 4),
            ("n6", "n7", 6 * 4),
            ("n7", "n8", 6 * 4),
            ("n1", "n8", size(frame, "uint8")),
            ("n8", "n9", size(frame, "uint8")),
        ],
    )


BENCHMARKS = (
    t01_object_detection,
    t02_sobel,
    t03_classification,
    t04_fir,
    t05_radar,
    t06_aes,
    t07_matmul,
    t08_jpeg,
    t09_kalman,
    t10_stabilization,
)


def main() -> None:
    for make in BENCHMARKS:
        task_file, tdg = make()
        write_json(HERE / "tasks" / f"{task_file['id']}.json", task_file)
        write_json(HERE / "reference" / f"{task_file['id']}.tdg.json", tdg)
        print(f"wrote {task_file['id']} ({len(tdg['nodes'])} nodes)")


if __name__ == "__main__":
    main()
