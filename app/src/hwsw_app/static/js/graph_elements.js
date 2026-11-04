// TDG -> the nodes and edges that the graph view draws. Pure functions (no DOM).

import { formatBytes, formatSeconds } from "./format.js";

export function buildElements(tdg) {
  const costs = new Map((tdg.costs?.nodes ?? []).map((cost) => [cost.id, cost]));
  const nodes = tdg.nodes.map((node) => ({
    data: {
      id: node.id,
      label: nodeLabel(node, costs.get(node.id)),
      affinity: node.affinity,
      phase: node.phase,
      kind: node.kind,
    },
  }));
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

/** The task name, plus its CPU time and best FPGA time once the TDG has been costed. */
export function nodeLabel(node, cost) {
  const lines = [node.name];
  const times = [];
  if (cost?.sw) times.push(`cpu ${formatSeconds(cost.sw.time_s.value)}`);
  if (cost?.hw?.length) {
    const fastest = Math.min(...cost.hw.map((point) => point.time_s.value));
    times.push(`fpga ${formatSeconds(fastest)}`);
  }
  if (times.length) lines.push(times.join(" · "));
  return lines.join("\n");
}
