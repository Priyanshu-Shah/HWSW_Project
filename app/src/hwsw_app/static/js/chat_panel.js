// The conversation: shows the agent's events and sends the user's turns.
// Event types are defined in pre_TDG/src/hwsw/conversation/events.py.

import { api, sendMessage } from "./api.js";
import { clear, el } from "./dom.js";

const GREETING = "Describe the task you want to partition, in your own words.";

export class ChatPanel {
  /** `onTdgReady(tdg)` is called when the agent hands over a task graph. */
  constructor({ messages, form, input, resetButton, onTdgReady }) {
    this.messages = messages;
    this.input = input;
    this.submitButton = form.querySelector("button[type=submit]");
    this.onTdgReady = onTdgReady;
    this.sessionId = null;

    form.addEventListener("submit", (event) => {
      event.preventDefault();
      this.#submitText();
    });
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        this.#submitText();
      }
    });
    resetButton.addEventListener("click", () => this.reset());
    this.reset();
  }

  reset() {
    this.sessionId = null;
    clear(this.messages);
    this.#bubble("assistant", GREETING);
  }

  #submitText() {
    const text = this.input.value.trim();
    if (!text || this.submitButton.disabled) return;
    this.input.value = "";
    this.#bubble("user", text);
    this.#send({ text });
  }

  async #send(userInput) {
    this.submitButton.disabled = true;
    try {
      this.sessionId ??= await api.createSession();
      for await (const { event, data } of sendMessage(this.sessionId, userInput)) {
        if (event === "end") break;
        this.#render(event, data);
      }
    } catch (failure) {
      this.#bubble("error", `Could not reach the server: ${failure.message}`);
    } finally {
      this.submitButton.disabled = false;
    }
  }

  #render(event, data) {
    switch (event) {
      case "assistant_message":
        this.#bubble("assistant", data.text);
        break;
      case "question_batch":
        this.#questionCard(data.questions);
        break;
      case "assumptions":
        this.#assumptionsCard(data.assumptions);
        break;
      case "spec_ready":
        this.#specCard(data.spec);
        break;
      case "tdg_ready":
        this.#bubble("assistant", `Task graph ready: ${data.tdg.nodes.length} tasks.`);
        this.onTdgReady(data.tdg);
        break;
      case "error":
        this.#bubble("error", data.message);
        break;
      default:
        this.#bubble("error", `Unknown event from the agent: ${event}`);
    }
  }

  #bubble(kind, text) {
    this.#append(el("div", { class: `bubble ${kind}` }, text));
  }

  #questionCard(questions) {
    const fields = questions.map((question) => ({ question, control: answerControl(question) }));
    const submit = el("button", { type: "submit", class: "primary" }, "Send answers");
    const card = el("form", { class: "card" },
      el("h3", {}, "A few questions"),
      ...fields.map(({ question, control }) =>
        el("label", {}, el("span", {}, question.text), control, choiceList(question, control))),
      submit,
    );
    card.addEventListener("submit", (event) => {
      event.preventDefault();
      const answers = Object.fromEntries(fields.map(({ question, control }) => [question.id, control.value]));
      fields.forEach(({ control }) => (control.disabled = true));
      submit.disabled = true;
      this.#bubble("user", fields.map(({ question }) => `${question.id}: ${answers[question.id]}`).join("\n"));
      this.#send({ answers });
    });
    this.#append(card);
  }

  #assumptionsCard(assumptions) {
    this.#append(el("div", { class: "card" },
      el("h3", {}, "What I am assuming"),
      el("table", {},
        el("tr", {}, ...["Item", "Value", "Source", "Confidence"].map((name) => el("th", {}, name))),
        ...assumptions.map((item) =>
          el("tr", {},
            el("td", {}, item.key),
            el("td", {}, item.value),
            el("td", {}, sourceCell(item)),
            el("td", {}, item.confidence))),
      ),
    ));
  }

  #specCard(spec) {
    this.#append(el("div", { class: "card" },
      el("details", {}, el("summary", {}, "Task specification"),
        el("pre", {}, JSON.stringify(spec, null, 2)))));
  }

  #append(node) {
    this.messages.append(node);
    this.messages.scrollTop = this.messages.scrollHeight;
  }
}

let choiceListCount = 0;

// A text box pre-filled with the suggestion. Choices are only offered: the agent's list can miss
// what the user wants (e.g. a newer model), so any other answer can be typed.
function answerControl(question) {
  const control = el("input", { type: "text" });
  control.value = question.suggested_answer;
  return control;
}

function choiceList(question, control) {
  if (!question.choices.length) return [];
  const id = `choices-${++choiceListCount}`;
  control.setAttribute("list", id);
  return el("datalist", { id }, ...question.choices.map((choice) => el("option", { value: choice })));
}

function sourceCell(item) {
  return item.url
    ? el("a", { href: item.url, target: "_blank", rel: "noopener" }, item.source)
    : el("span", { class: "badge" }, item.source);
}
