import assert from "node:assert/strict";
import { test } from "node:test";

import { parseSseMessage, splitSseMessages } from "../../src/hwsw_app/static/js/sse.js";

test("parses one message", () => {
  const message = parseSseMessage('event: assistant_message\ndata: {"text":"hi"}');
  assert.deepEqual(message, { event: "assistant_message", data: { text: "hi" } });
});

test("keeps an unfinished tail for the next chunk", () => {
  const first = splitSseMessages('event: a\ndata: {"n":1}\n\nevent: b\ndata: {"n"');
  assert.equal(first.messages.length, 1);
  assert.equal(first.messages[0].event, "a");
  assert.equal(first.rest, 'event: b\ndata: {"n"');

  const second = splitSseMessages(first.rest + ":2}\n\n");
  assert.deepEqual(second.messages, [{ event: "b", data: { n: 2 } }]);
  assert.equal(second.rest, "");
});

test("several messages in one chunk, including the end marker", () => {
  const { messages } = splitSseMessages("event: a\ndata: {}\n\nevent: end\ndata: {}\n\n");
  assert.deepEqual(messages.map((message) => message.event), ["a", "end"]);
});
