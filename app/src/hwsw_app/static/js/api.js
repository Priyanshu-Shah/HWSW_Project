// Calls to the server.

import { splitSseMessages } from "./sse.js";

async function getJson(url) {
  return readJson(await fetch(url));
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return readJson(response);
}

async function readJson(response) {
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${detail}`);
  }
  return response.json();
}

export const api = {
  boards: () => getJson("/api/boards"),
  exampleNames: () => getJson("/api/examples").then((body) => body.names),
  example: (name) => getJson(`/api/examples/${encodeURIComponent(name)}`),
  validate: (tdg) => postJson("/api/tdg/validate", { tdg }),
  cost: (tdg) => postJson("/api/tdg/cost", { tdg }),
  createSession: () => postJson("/api/chat/sessions", {}).then((body) => body.session_id),
};

/**
 * Send one user turn and yield the agent's events as they arrive.
 * Each yielded value is {event, data}; the stream ends with event "end".
 */
export async function* sendMessage(sessionId, userInput) {
  const response = await fetch(`/api/chat/sessions/${sessionId}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(userInput),
  });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const { messages, rest } = splitSseMessages(buffer);
    buffer = rest;
    for (const message of messages) yield message;
  }
}
