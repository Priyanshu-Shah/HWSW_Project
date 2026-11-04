// A just-enough DOM for testing the modules that build elements, without a browser.
// `append` behaves like the real one: anything that is not a node becomes text via String(),
// which is how an accidentally passed array shows up as "[object ...]".

class FakeNode {
  constructor(tag) {
    this.tag = tag;
    this.children = [];
    this.attributes = {};
    this.className = "";
  }
  append(...items) {
    for (const item of items) this.children.push(item instanceof FakeNode ? item : { text: String(item) });
  }
  replaceChildren(...items) {
    this.children = [];
    this.append(...items);
  }
  setAttribute(name, value) {
    this.attributes[name] = value;
  }
  addEventListener() {}
  get text() {
    return this.children.map((child) => (child instanceof FakeNode ? child.text : child.text)).join(" ");
  }
}

export function installFakeDocument() {
  globalThis.document = { createElement: (tag) => new FakeNode(tag) };
}

export { FakeNode };
