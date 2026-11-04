// Wires the page together: toolbar, tabs, results panels.

import { api } from "./api.js";
import { ChatPanel } from "./chat_panel.js";
import { clear, el } from "./dom.js";
import { formatSeconds } from "./format.js";
import { GraphView } from "./graph_view.js";
import { renderNodeDetails } from "./node_details.js";
import { enableSplitter } from "./splitter.js";

const $ = (id) => document.getElementById(id);
const state = { tdg: null };

const graphView = new GraphView($("graph"), selectNode);
new ChatPanel({
  messages: $("messages"),
  form: $("composer"),
  input: $("composer-input"),
  resetButton: $("new-conversation"),
  onTdgReady: (tdg) => showTdg(tdg),
});

enableSplitter({
  axis: "height",
  handle: $("details-splitter"),
  panel: $("details"),
  container: $("tab-graph"),
  onResize: () => graphView.refresh(),
});
enableSplitter({
  axis: "width",
  handle: $("chat-splitter"),
  panel: $("chat"),
  container: document.querySelector("main"),
  onResize: () => graphView.refresh(),
});

// ---- showing a TDG ----------------------------------------------------------------------

/** Make `tdg` the current graph. `keepResults` is true when it is the costed version of the same graph. */
function showTdg(tdg, { keepResults = false } = {}) {
  state.tdg = tdg;
  graphView.show(tdg);
  $("empty-graph").hidden = true;
  $("json-view").textContent = JSON.stringify(tdg, null, 2);
  for (const id of ["validate", "cost", "download"]) $(id).disabled = false;
  selectNode(null);
  if (!keepResults) {
    setStatus("");
    renderBounds(null);
    renderIssues([]);
  }
  switchTab("graph");
}

function selectNode(nodeId) {
  const node = state.tdg?.nodes.find((candidate) => candidate.id === nodeId);
  const cost = state.tdg?.costs?.nodes.find((candidate) => candidate.id === nodeId);
  renderNodeDetails($("details"), node, cost);
}

// ---- toolbar ----------------------------------------------------------------------------

$("load-example").addEventListener("click", () =>
  guarded(async () => showTdg(await api.example($("example-select").value))));

$("validate").addEventListener("click", () =>
  guarded(async () => {
    const result = await api.validate(state.tdg);
    renderIssues(result.issues);
    setStatus(result.valid ? "Valid." : "Not valid.", result.valid ? "ok" : "error");
    if (!result.valid) switchTab("issues");
  }));

$("cost").addEventListener("click", () =>
  guarded(async () => {
    const result = await api.cost(state.tdg);
    renderIssues(result.issues);
    if (result.costed_tdg) showTdg(result.costed_tdg, { keepResults: true });
    renderBounds(result.bounds);
    if (result.valid) {
      setStatus("Costed.", "ok");
    } else {
      setStatus("Costing found problems.", "error");
      switchTab("issues");
    }
  }));

$("download").addEventListener("click", () => {
  const blob = new Blob([JSON.stringify(state.tdg, null, 2)], { type: "application/json" });
  const link = el("a", { href: URL.createObjectURL(blob), download: `${state.tdg.graph.id}.tdg.json` });
  link.click();
  URL.revokeObjectURL(link.href);
});

async function guarded(action) {
  setStatus("Working…");
  try {
    await action();
  } catch (failure) {
    setStatus(failure.message, "error");
  }
}

function setStatus(text, kind = "") {
  $("status").textContent = text;
  $("status").className = kind;
}

// ---- tabs and result panels -------------------------------------------------------------

for (const button of document.querySelectorAll("#tabs button")) {
  button.addEventListener("click", () => switchTab(button.dataset.tab));
}

function switchTab(name) {
  for (const button of document.querySelectorAll("#tabs button")) {
    button.classList.toggle("active", button.dataset.tab === name);
  }
  for (const tab of document.querySelectorAll(".tab")) tab.hidden = tab.id !== `tab-${name}`;
  if (name === "graph") graphView.refresh();
}

function renderIssues(issues) {
  const list = $("issues");
  clear(list);
  if (!issues.length) list.append(el("li", { class: "muted" }, "No issues."));
  for (const issue of issues) {
    list.append(el("li", {},
      el("span", { class: `sev-${issue.severity}` }, issue.severity),
      ` [${issue.rule}] ${issue.location}: ${issue.message}`));
  }
  $("issue-count").textContent = String(issues.length);
  $("issue-count").hidden = issues.length === 0;
}

function renderBounds(bounds) {
  const panel = $("bounds");
  panel.hidden = bounds === null;
  if (bounds === null) return;
  clear(panel);
  const allSoftware = bounds.all_software_s === null ? "n/a" : formatSeconds(bounds.all_software_s);
  const deadline = bounds.deadline_s === null ? "none" : formatSeconds(bounds.deadline_s);
  panel.append(
    el("div", {},
      el("strong", {}, "Latency bounds: "),
      `all on CPU ${allSoftware} · best case ${formatSeconds(bounds.best_case_s)} `,
      `(optimistic ${formatSeconds(bounds.best_case_optimistic_s)}) · deadline ${deadline}`),
  );
  if (bounds.warnings.length) {
    panel.append(el("ul", {}, ...bounds.warnings.map((warning) => el("li", {}, warning))));
  }
  panel.classList.toggle("warn", bounds.warnings.length > 0);
}

// ---- start-up ---------------------------------------------------------------------------

async function init() {
  try {
    const { boards, source } = await api.boards();
    const chip = $("boards-chip");
    chip.textContent = `${boards.length} boards · ${source}`;
    if (source === "fixtures") {
      chip.classList.add("warn");
      chip.title = "library/boards is empty, so test fixture boards are used. Their numbers are not authoritative.";
    }
    const names = await api.exampleNames();
    $("example-select").replaceChildren(...names.map((name) => el("option", { value: name }, name)));
  } catch (failure) {
    setStatus(`Could not load boards and examples: ${failure.message}`, "error");
  }
}

init();
