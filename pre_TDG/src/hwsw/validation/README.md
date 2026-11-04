# validation

**What it does in one sentence:** checks that a TDG file is well-formed and makes sense, and says
exactly what is wrong in a sentence that an LLM (or a person) can act on.

It is used in three places: the LLM TDG builder (to get errors to fix, then retry), `hwsw cost`
(to refuse bad input and check its own output), and `hwsw validate` (by hand).

## The intuition

Two kinds of mistakes can be in a TDG:

1. **Wrong shape**: a field is missing, has the wrong type, a value is not allowed (`dtype:
   "int4"`). A JSON Schema can describe all of this, so the contract file
   `TDG/schemas/tdg.schema.json` does it. This check runs **first**, and if it fails we stop:
   the other checks assume the shape is right.
2. **Wrong meaning**: every field is fine on its own, but together they do not make sense (an
   edge points to a node that does not exist, the graph has a loop, an FPGA design uses more
   DSPs than the board has). A schema cannot say this, so there is a small function for each rule.

```text
validate_tdg(tdg, board)            tdg_validator.py
  1. json_schema_rules.py           shape (stops here if anything is wrong)
  2. graph_rules.py                 R1 R2 R3
     loop_nest_rules.py             R12
     constraint_rules.py            R10 R11
     cost_map_rules.py   (only if the TDG has `costs`)    R1 R2 R4 R6 R8 R9
     platform_rules.py   (only if a board is given)       R5
  -> ValidationReport: a list of Issues
```

Each problem is an `Issue(rule, severity, location, message)`:

```text
error [R2] edge e1: dst 'n9' does not exist; existing node ids: n1, n2, n3
error [SCHEMA] nodes[2].template.params: 'C_out' is a required property
warning [R11] constraints.deadline_s: deadline 0.05 s is longer than the period 0.0333 s; ...
```

- **error**: the document must not be passed on. `report.is_valid` is false.
- **warning**: suspicious, but allowed through (for example R9, R11).
- Messages name the place and the allowed alternatives, so a repair prompt can be as simple as
  "here are the errors, fix the TDG".

## The rules

The numbering R1-R12 is defined in the header of `TDG/schemas/tdg.schema.json` (the contract).

| Rule | Checks | Severity | File |
|---|---|---|---|
| SCHEMA | shape, types, allowed values, the right payload for each node `kind` | error | `json_schema_rules.py` |
| R1 | node ids unique, edge ids unique, one cost entry per node, hardware point ids unique | error | `graph_rules.py`, `cost_map_rules.py` |
| R2 | edge ends, and cost-map entries, refer to things that exist | error | `graph_rules.py`, `cost_map_rules.py` |
| R3 | no cycles (data and control edges together) | error | `graph_rules.py` |
| R4 | every estimate has `lo <= value <= hi` | error | `cost_map_rules.py` |
| R5 | the board exists; every hardware point fits its resources; edge interface overrides exist | error | `platform_rules.py` |
| R6 | a `hw_only` node has at least one hardware point | error | `cost_map_rules.py` |
| R7 | (a rule for the consumer: an opaque node with no hardware point is software-only) | not checked | |
| R8 | every `iteration` node that is not `hw_only` has a software cost | error | `cost_map_rules.py` |
| R9 | `ops_estimated` within 2x of the LLM's `ops_claimed` | warning | `cost_map_rules.py` |
| R10 | the objective is exactly `{latency: 1.0}` | error | `constraint_rules.py` |
| R11 | `deadline_s` is not longer than `period_s` | warning | `constraint_rules.py` |
| R12 | loop-nest names are consistent: unique loop variables; `reuse`, `unrollable` and `access` refer to declared loops/arrays | error | `loop_nest_rules.py` |

The schema check turns the schema's own `oneOf` errors into readable ones in
`json_schema_rules.py` (`_most_specific`): a `loop_nest` node would otherwise also be reported as
"not an opaque node" and "not a template node", which is noise.

## Other files

- `issue.py`: `Issue`, `Severity`, `ValidationReport`.
- `id_listing.py`: shortens long id lists in messages.
- `../spec/`: the same machinery checks the internal task spec (`validate_spec`).

## Adding or changing a rule

1. If it is a contract rule (R1-R12 or a schema change): **stop and tell the user first**, because
   other teams read the same contract (`TDG/decisions.md` D5).
2. Otherwise add a function `check_xxx(tdg) -> list[Issue]` in the matching `*_rules.py` file
   (or a new one named after the rule group) and call it from `tdg_validator.py`.
3. Write the message so it states what is wrong **and** what is allowed or where to look.
4. Add a test in `tests/validation/` that triggers only that rule and checks the message.

## TODO

- [ ] Edge `bytes` against the loop nests' array sizes (a mismatch is a likely LLM mistake)
- [ ] Check template parameters (for example conv `groups`, kernel larger than the input) here
      instead of only when costing, so the repair loop sees them
- [ ] Warn about nodes with no edges (disconnected from the rest of the graph) and about
      duplicate node names
- [ ] Spec against TDG: the TDG's `platform`, `arrival` and `deadline_s` match the spec
- [ ] Validate board files against the library team's `platform.schema.json` once it exists
- [ ] Decide with the user whether R7 should become a check (needs the partitioner team's view)
