"""R12: names used inside a loop-nest description must refer to things it declares."""

from collections import Counter

from hwsw.validation.issue import Issue, error


def check_loop_nests(tdg: dict) -> list[Issue]:
    issues = []
    for node in tdg["nodes"]:
        if node["kind"] == "loop_nest":
            issues += _check_one_loop_nest(node["id"], node["loop_nest"])
    return issues


def _check_one_loop_nest(node_id: str, loop_nest: dict) -> list[Issue]:
    location = f"node {node_id}"
    loop_vars = [loop["var"] for loop in loop_nest["loops"]]
    issues = [
        error("R12", location, f"loop variable '{var}' is declared {count} times")
        for var, count in Counter(loop_vars).items()
        if count > 1
    ]

    for array_name, access in loop_nest.get("access", {}).items():
        if array_name not in loop_nest["arrays"]:
            issues.append(
                error(
                    "R12",
                    location,
                    f"access describes array '{array_name}', which is not in arrays",
                )
            )
        for var in access.get("reuse", []):
            if var not in loop_vars:
                issues.append(
                    error(
                        "R12",
                        location,
                        f"access.{array_name}.reuse uses unknown loop variable '{var}'",
                    )
                )

    for var in loop_nest.get("hints", {}).get("unrollable", []):
        if var not in loop_vars:
            issues.append(
                error("R12", location, f"hints.unrollable uses unknown loop variable '{var}'")
            )
    return issues
