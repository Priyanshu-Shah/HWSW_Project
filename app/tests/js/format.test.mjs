import assert from "node:assert/strict";
import { test } from "node:test";

import { formatBytes, formatCount, formatSeconds } from "../../src/hwsw_app/static/js/format.js";

test("bytes", () => {
  assert.equal(formatBytes(512), "512 B");
  assert.equal(formatBytes(2764800), "2.6 MB");
  assert.equal(formatBytes(1536), "1.5 KB");
  assert.equal(formatBytes(5 * 1024 ** 3), "5.0 GB");
});

test("seconds pick a sensible unit", () => {
  assert.equal(formatSeconds(2), "2.00 s");
  assert.equal(formatSeconds(0.0209), "20.9 ms");
  assert.equal(formatSeconds(0.0000125), "12.5 µs");
  assert.equal(formatSeconds(3e-9), "3.00 ns");
  assert.equal(formatSeconds(null), "–");
});

test("counts get thousands separators", () => {
  assert.equal(formatCount(37675044), "37,675,044");
});
