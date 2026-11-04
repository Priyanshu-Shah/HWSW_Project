// Drag handles that resize a panel: the details panel (height) and the conversation pane (width).
// Drag, or focus the handle and use the arrow keys; double-click resets to the default size.

const KEY_STEP = 24;

// `direction` is how the panel's size changes when the pointer moves in the positive direction:
// the details panel sits below its handle, so dragging up (negative) makes it bigger; the
// conversation pane sits left of its handle, so dragging right (positive) makes it bigger.
const AXES = {
  height: {
    pointer: "clientY", direction: -1, min: 80, reserve: 200,
    available: (container) => container.clientHeight,
    size: (panel) => panel.getBoundingClientRect().height,
    grow: "ArrowUp", shrink: "ArrowDown", storageKey: "hwsw.detailsHeight",
  },
  width: {
    pointer: "clientX", direction: 1, min: 260, reserve: 520,
    available: (container) => container.clientWidth,
    size: (panel) => panel.getBoundingClientRect().width,
    grow: "ArrowRight", shrink: "ArrowLeft", storageKey: "hwsw.chatWidth",
  },
}; // fmt: skip

/** `size` limited to [min, max]; if the range is empty, `min` wins. */
export function clampSize(size, min, max) {
  return Math.min(Math.max(size, min), Math.max(min, max));
}

/**
 * `axis` is "height" or "width". `handle` is the grab bar, `panel` the element that changes
 * size, `container` the element holding the handle, the panel and the space that must remain
 * for the rest; `onResize` runs after the size changed (to refit the graph).
 */
export function enableSplitter({ axis, handle, panel, container, onResize }) {
  const config = AXES[axis];
  let resizePending = false;

  const maxSize = () => config.available(container) - config.reserve;

  function setSize(size, { remember = true } = {}) {
    const clamped = clampSize(size, config.min, maxSize());
    panel.style[axis] = `${clamped}px`;
    if (remember) store(config.storageKey, clamped);
    if (!resizePending) {
      resizePending = true;
      requestAnimationFrame(() => {
        resizePending = false;
        onResize();
      });
    }
  }

  let drag = null;
  handle.addEventListener("pointerdown", (event) => {
    drag = { start: event[config.pointer], startSize: config.size(panel) };
    handle.setPointerCapture(event.pointerId);
    handle.classList.add("dragging");
    document.body.style.userSelect = "none";
  });
  handle.addEventListener("pointermove", (event) => {
    if (drag) setSize(drag.startSize + config.direction * (event[config.pointer] - drag.start));
  });
  const endDrag = () => {
    drag = null;
    handle.classList.remove("dragging");
    document.body.style.userSelect = "";
  };
  handle.addEventListener("pointerup", endDrag);
  handle.addEventListener("pointercancel", endDrag);

  handle.addEventListener("keydown", (event) => {
    if (event.key === config.grow) setSize(config.size(panel) + KEY_STEP);
    else if (event.key === config.shrink) setSize(config.size(panel) - KEY_STEP);
    else return;
    event.preventDefault();
  });

  handle.addEventListener("dblclick", () => {
    panel.style[axis] = "";
    forget(config.storageKey);
    onResize();
  });

  window.addEventListener("resize", () => {
    if (panel.style[axis]) setSize(config.size(panel), { remember: false });
  });

  const saved = Number(read(config.storageKey));
  if (saved) setSize(saved, { remember: false });
}

// localStorage can be unavailable (private windows, blocked site data): resizing still works.
function read(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function store(key, size) {
  try {
    localStorage.setItem(key, String(Math.round(size)));
  } catch {
    /* not remembered */
  }
}

function forget(key) {
  try {
    localStorage.removeItem(key);
  } catch {
    /* nothing to forget */
  }
}
