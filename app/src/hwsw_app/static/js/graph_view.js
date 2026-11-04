// Draws a TDG as a left-to-right graph (Cytoscape.js + dagre layout, loaded as globals).

import { buildElements } from "./graph_elements.js";

const FIT_PADDING = 30;
const MAX_FIT_ZOOM = 1.25; // a small graph should not be blown up to fill the panel

export class GraphView {
  /** `onSelect(nodeId | null)` is called when the user clicks a task or the background. */
  constructor(container, onSelect) {
    this.cy = cytoscape({
      container,
      style: graphStyle(),
      wheelSensitivity: 0.3,
      minZoom: 0.2,
      maxZoom: 3,
    });
    this.cy.on("tap", "node", (event) => onSelect(event.target.id()));
    this.cy.on("tap", (event) => {
      if (event.target === this.cy) onSelect(null);
    });
  }

  show(tdg) {
    const { nodes, edges } = buildElements(tdg);
    this.cy.elements().remove();
    this.cy.add([...nodes, ...edges]);
    this.cy
      .layout({ name: "dagre", rankDir: "LR", nodeSep: 30, rankSep: 70, animate: false })
      .run();
    this.refresh();
  }

  /** Call after the container changes size or becomes visible. */
  refresh() {
    this.cy.resize();
    this.cy.fit(undefined, FIT_PADDING);
    if (this.cy.zoom() > MAX_FIT_ZOOM) {
      this.cy.zoom(MAX_FIT_ZOOM);
      this.cy.center();
    }
  }
}

function graphStyle() {
  const css = getComputedStyle(document.documentElement);
  const color = (name) => css.getPropertyValue(name).trim();
  return [
    {
      selector: "node",
      style: {
        shape: "round-rectangle",
        label: "data(label)",
        "text-wrap": "wrap",
        "text-valign": "center",
        "text-halign": "center",
        "font-size": 12,
        color: color("--text"),
        width: "label",
        height: "label",
        padding: "10px",
        "background-color": color("--node-any"),
        "border-width": 1,
        "border-color": color("--node-border"),
      },
    },
    { selector: 'node[affinity = "sw_only"]', style: { "background-color": color("--node-sw") } },
    { selector: 'node[affinity = "hw_only"]', style: { "background-color": color("--node-hw") } },
    { selector: 'node[phase = "setup"]', style: { "border-style": "dashed" } },
    {
      selector: "node:selected",
      style: { "border-width": 3, "border-color": color("--accent") },
    },
    {
      selector: "edge",
      style: {
        width: 1.5,
        "curve-style": "bezier",
        "line-color": color("--edge"),
        "target-arrow-color": color("--edge"),
        "target-arrow-shape": "triangle",
        label: "data(label)",
        "font-size": 10,
        color: color("--muted"),
        "text-background-color": color("--panel"),
        "text-background-opacity": 1,
        "text-background-padding": "2px",
      },
    },
    { selector: 'edge[kind = "control"]', style: { "line-style": "dashed" } },
  ];
}
