// The "details" panel: what one task is, and what it costs once the TDG has been costed.

import { clear, el } from "./dom.js";
import { formatCount, formatSeconds } from "./format.js";

export function renderNodeDetails(container, node, cost) {
  clear(container);
  if (!node) {
    container.append(el("p", { class: "muted" }, "Click a task to see its details."));
    return;
  }
  container.append(
    el("h3", {}, node.name),
    table([
      ["Id", node.id],
      ["Phase", node.phase === "setup" ? "setup (runs once)" : "iteration (every run)"],
      ["Where it may run", node.affinity],
      ["What it is", describeKind(node)],
      node.ops_claimed === undefined
        ? null
        : ["Operations (LLM's estimate)", formatCount(node.ops_claimed)],
    ]),
    cost ? costDetails(cost) : el("p", { class: "muted" }, "Not costed yet. Press “Cost it”."),
  );
}

function describeKind(node) {
  if (node.kind === "opaque") {
    return `fixed ${formatSeconds(node.opaque.sw_time_s)} on the CPU (${node.opaque.source})`;
  }
  if (node.kind === "template") {
    return `${node.template.op} ${JSON.stringify(node.template.params)}`;
  }
  const loops = node.loop_nest.loops.map((loop) => `${loop.var}:${loop.range}`).join(" × ");
  return `loop nest ${loops}, per iteration ${JSON.stringify(node.loop_nest.body)}`;
}

function costDetails(cost) {
  const parts = [];
  const rows = [];
  if (cost.ops_estimated !== undefined) {
    rows.push(["Operations (estimated)", formatCount(cost.ops_estimated)]);
  }
  if (cost.sw) rows.push(["CPU time", withRange(cost.sw.time_s)]);
  parts.push(table(rows));

  if (cost.hw?.length) {
    parts.push(
      el("table", {},
        el("tr", {}, ...["FPGA option", "Parallelism", "Time", "LUT", "FF", "DSP", "BRAM18K"]
          .map((heading) => el("th", {}, heading))),
        ...cost.hw.map((point) =>
          el("tr", {},
            ...[
              point.point,
              point.parallelism ?? "–",
              withRange(point.time_s),
              formatCount(point.resources.lut ?? 0),
              formatCount(point.resources.ff ?? 0),
              formatCount(point.resources.dsp ?? 0),
              formatCount(point.resources.bram18k ?? 0),
            ].map((value) => el("td", {}, String(value))))),
      ),
    );
  } else {
    parts.push(el("p", { class: "muted" }, "No FPGA option (CPU only, or nothing fits the board)."));
  }
  return el("div", {}, ...parts);
}

function withRange(estimate) {
  const { value, lo, hi } = estimate;
  return `${formatSeconds(value)} (${formatSeconds(lo)} – ${formatSeconds(hi)})`;
}

function table(rows) {
  return el("table", {},
    ...rows.filter(Boolean).map(([name, value]) =>
      el("tr", {}, el("th", {}, name), el("td", {}, String(value)))));
}
