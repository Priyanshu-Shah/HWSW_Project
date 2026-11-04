// Small DOM helper. Text always goes in as text nodes, never as HTML: task names and
// assumptions come from an LLM or from user input.

/** el("div", {class: "card", onclick: handler}, "text", childElement, ...) */
export function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(props)) {
    if (value === false || value === null || value === undefined) continue;
    if (name === "class") node.className = value;
    else if (name.startsWith("on")) node.addEventListener(name.slice(2), value);
    else node.setAttribute(name, value === true ? "" : value);
  }
  for (const child of children.flat(Infinity)) node.append(child);
  return node;
}

export function clear(node) {
  node.replaceChildren();
}
