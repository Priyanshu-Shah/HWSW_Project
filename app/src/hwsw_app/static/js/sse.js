// Parsing of server-sent events from a streamed response. Pure functions.

/** Split a text buffer into complete messages; `rest` is the unfinished tail. */
export function splitSseMessages(buffer) {
  const parts = buffer.split("\n\n");
  const rest = parts.pop();
  return { messages: parts.filter(Boolean).map(parseSseMessage), rest };
}

export function parseSseMessage(text) {
  let event = "message";
  const dataLines = [];
  for (const line of text.split("\n")) {
    if (line.startsWith("event:")) event = line.slice("event:".length).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice("data:".length).trimStart());
  }
  return { event, data: JSON.parse(dataLines.join("\n") || "null") };
}
