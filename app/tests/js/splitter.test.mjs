import assert from "node:assert/strict";
import { test } from "node:test";

import { clampSize } from "../../src/hwsw_app/static/js/splitter.js";

test("a height inside the range is kept", () => {
  assert.equal(clampSize(150, 80, 400), 150);
});

test("too small or too large heights are limited", () => {
  assert.equal(clampSize(10, 80, 400), 80);
  assert.equal(clampSize(900, 80, 400), 400);
});

test("when the window is too short for the maximum, the minimum still wins", () => {
  assert.equal(clampSize(300, 80, 50), 80);
});
