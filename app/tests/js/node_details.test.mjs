import assert from "node:assert/strict";
import { test } from "node:test";

import { installFakeDocument } from "./fake_dom.mjs";

installFakeDocument();
const { renderNodeDetails } = await import("../../src/hwsw_app/static/js/node_details.js");
const { el } = await import("../../src/hwsw_app/static/js/dom.js");

const estimate = (value) => ({ value, lo: value / 2, hi: value * 2, source: "roofline" });

const node = {
  id: "n2", name: "resize", phase: "iteration", affinity: "any", kind: "loop_nest",
  loop_nest: { loops: [{ var: "y", range: 4 }, { var: "x", range: 8 }], body: { mac: 1 } },
  ops_claimed: 1000,
};
const cost = {
  id: "n2",
  ops_estimated: 1024,
  sw: { time_s: estimate(0.002) },
  hw: [{
    point: "p1", parallelism: 4, time_s: estimate(0.001), ii_s: 0.001,
    resources: { lut: 1644, ff: 2466, dsp: 9, bram18k: 4 },
  }],
};

function render(taskNode, taskCost) {
  const container = el("div");
  renderNodeDetails(container, taskNode, taskCost);
  return container.text;
}

test("a costed task shows its CPU time and FPGA options", () => {
  const text = render(node, cost);
  assert.match(text, /CPU time/);
  assert.match(text, /2\.00 ms/);
  assert.match(text, /FPGA option/);
  assert.match(text, /p1/);
  assert.match(text, /1,644/);
});

test("nothing is ever shown as an [object ...] placeholder", () => {
  for (const text of [render(node, cost), render(node, undefined), render(null, undefined)]) {
    assert.doesNotMatch(text, /\[object/);
  }
});

test("an uncosted task says so", () => {
  assert.match(render(node, undefined), /Not costed yet/);
});

test("a task with no FPGA option says so", () => {
  assert.match(render(node, { ...cost, hw: [] }), /No FPGA option/);
});

test("el flattens nested child lists instead of stringifying them", () => {
  const parent = el("div", {}, [el("span"), [el("b")]]);
  assert.equal(parent.children.length, 2);
  assert.ok(parent.children.every((child) => child.tag));
});
