// TDG -> the nodes and edges that the graph view draws. Pure functions (no DOM).

import { formatBytes, formatSeconds } from "./format.js";

/** `partition` (optional) is post_TDG's result: each node is then coloured by where it runs. */
export function buildElements(tdg, partition = null) {
  const costs = new Map((tdg.costs?.nodes ?? []).map((cost) => [cost.id, cost]));
  const placed = new Map((partition?.nodes ?? []).map((item) => [item.id, item]));
  const nodes = tdg.nodes.map((node) => {
    const assignment = placed.get(node.id) ?? null;
    const data = {
      id: node.id,
      label: nodeLabel(node, costs.get(node.id), assignment),
      affinity: node.affinity,
      phase: node.phase,
      kind: node.kind,
    };
    if (assignment) data.target = assignment.target;
    return { data };
  });
  const edges = tdg.edges.map((edge) => ({
    data: {
      id: edge.id,
      source: edge.src,
      target: edge.dst,
      kind: edge.kind,
      label: edge.kind === "control" ? "control" : formatBytes(edge.bytes),
    },
  }));
  return { nodes, edges };
}

/** The task name, its CPU time and best FPGA time once costed, and where it runs once partitioned. */
export function nodeLabel(node, cost, assignment = null) {
  const lines = [node.name];
  const times = [];
  if (cost?.sw) times.push(`cpu ${formatSeconds(cost.sw.time_s.value)}`);
  if (cost?.hw?.length) {
    const fastest = Math.min(...cost.hw.map((point) => point.time_s.value));
    times.push(`fpga ${formatSeconds(fastest)}`);
  }
  if (times.length) lines.push(times.join(" · "));
  if (assignment) lines.push(assignmentText(assignment));
  return lines.join("\n");
}

export function assignmentText(assignment) {
  return assignment.target === "hw" ? `→ FPGA (${assignment.point})` : `→ CPU (${assignment.pe})`;
}
