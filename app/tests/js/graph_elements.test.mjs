import assert from "node:assert/strict";
import { test } from "node:test";

import { buildElements, nodeLabel } from "../../src/hwsw_app/static/js/graph_elements.js";

const graph = {
  nodes: [
    { id: "n1", name: "capture", phase: "iteration", affinity: "sw_only", kind: "opaque" },
    { id: "n2", name: "filter", phase: "setup", affinity: "any", kind: "template" },
  ],
  edges: [
    { id: "e1", src: "n1", dst: "n2", kind: "data", bytes: 2764800 },
    { id: "e2", src: "n1", dst: "n2", kind: "control", bytes: 0 },
  ],
};

const estimate = (value) => ({ value, lo: value / 2, hi: value * 2, source: "roofline" });

test("nodes and edges become graph elements", () => {
  const { nodes, edges } = buildElements(graph);
  assert.deepEqual(nodes.map((node) => node.data.id), ["n1", "n2"]);
  assert.equal(nodes[1].data.phase, "setup");
  assert.deepEqual(edges[0].data, {
    id: "e1", source: "n1", target: "n2", kind: "data", label: "2.6 MB",
  });
  assert.equal(edges[1].data.label, "control");
});

test("a graph without costs shows only the task names", () => {
  assert.equal(buildElements(graph).nodes[0].data.label, "capture");
});

test("costed nodes show the CPU time and the fastest FPGA option", () => {
  const costed = {
    ...graph,
    costs: {
      nodes: [
        { id: "n1", sw: { time_s: estimate(0.002) } },
        {
          id: "n2",
          sw: { time_s: estimate(0.0031) },
          hw: [{ time_s: estimate(0.0012) }, { time_s: estimate(0.0004) }],
        },
      ],
    },
  };
  const { nodes } = buildElements(costed);
  assert.equal(nodes[0].data.label, "capture\ncpu 2.00 ms");
  assert.equal(nodes[1].data.label, "filter\ncpu 3.10 ms · fpga 400 µs");
});

test("nodeLabel copes with a cost that has no hardware options", () => {
  const label = nodeLabel(graph.nodes[0], { id: "n1", sw: { time_s: estimate(0.5) }, hw: [] });
  assert.equal(label, "capture\ncpu 500 ms");
});
