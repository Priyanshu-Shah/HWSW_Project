"""`partition <costed TDG> --board <board file>`: writes `<name>.partition.json` and prints a summary.

Exit code 0 when a partition meets every constraint, 1 when none can (fb1) or none was found (fb2),
2 when the input is wrong, 3 on an unexpected error (traceback on stderr).
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from pydantic import ValidationError

from post_tdg.convert import ConversionError
from post_tdg.partition import DEFAULT_GNN, DEFAULT_TIME_LIMIT_S, partition


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tdg", type=Path, help="costed TDG JSON (graph + `costs`)")
    parser.add_argument("--board", type=Path, required=True, help="board file (library format)")
    parser.add_argument("--out", type=Path, help="result file (default: <tdg name>.partition.json)")
    parser.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT_S, help="solver budget, seconds")
    parser.add_argument("--no-gnn", action="store_true", help="solve without the GNN warm start")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        tdg = json.loads(args.tdg.read_text(encoding="utf-8"))
        board = json.loads(args.board.read_text(encoding="utf-8"))
        found = partition(
            tdg, board, time_limit_s=args.time_limit, gnn_path=None if args.no_gnn else DEFAULT_GNN, seed=args.seed
        )
    except (OSError, json.JSONDecodeError, ConversionError, KeyError, ValidationError) as failure:
        print(f"error: {failure}", file=sys.stderr)
        return 2
    except Exception:  # noqa: BLE001 - keep exit 1 meaning "answered, constraints not met"
        traceback.print_exc()
        return 3
    out = args.out or args.tdg.with_name(args.tdg.name.split(".")[0] + ".partition.json")
    out.write_text(json.dumps(found.result, indent=1), encoding="utf-8")
    print(summary(found.result))
    print(f"wrote {out}")
    return 0 if found.result["feasible"] else 1


def summary(result: dict) -> str:
    solver = result["solver"]
    head = f"{result['graph_id']} on {result['platform']}: {solver['method']} {solver['status']}"
    lines = [f"{head} ({solver['runtime_s']:.2f} s)"]
    latency = result["latency_s"]
    if latency:
        line = f"  latency {latency['nominal'] * 1e3:.3f} ms [p05 {latency['p05'] * 1e3:.3f}, p95 {latency['p95'] * 1e3:.3f}]"
        if result["all_software_s"]:
            line += f", all-software {result['all_software_s'] * 1e3:.3f} ms, speed-up {result['speedup']:.2f}x"
        lines.append(line)
    if result["deadline_s"]:
        lines.append(f"  deadline {result['deadline_s'] * 1e3:.3f} ms: {'met' if result['feasible'] else 'NOT met'}")
    for node in result["nodes"]:
        where = f"HW ({node['point']})" if node["target"] == "hw" else f"SW ({node['pe']})"
        lines.append(f"  {node['id']:5} {node['name'][:32]:32} {where}")
    for reason in result["feedback"]["reasons"]:
        lines.append(f"  [{result['feedback']['loop']}] {reason}")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
