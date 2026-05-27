import { describe, expect, it } from "vitest";

import {
  assistantMessagesReducer,
  computeLatency,
  createMessage,
  createSessionId,
  createSpeakerId,
} from "./assistantSession";

describe("assistant session state", () => {
  it("prepends messages and marks feedback", () => {
    const first = createMessage({ sender: "user", text: "오디스" });
    const second = createMessage({ sender: "odiss", text: "네, 말씀하세요." });

    let messages = assistantMessagesReducer([], { type: "append", message: first });
    messages = assistantMessagesReducer(messages, { type: "append", message: second });
    messages = assistantMessagesReducer(messages, { type: "feedback", messageId: second.id, rating: "up" });

    expect(messages[0].id).toBe(second.id);
    expect(messages[0].feedback).toBe("up");
    expect(messages[1].id).toBe(first.id);
  });

  it("computes latency only from valid timestamps", () => {
    expect(computeLatency({ wsSend: 100, firstMessage: 180, finalMessage: 260 })).toEqual({
      sttMs: undefined,
      firstMessageMs: 80,
      finalResponseMs: 160,
      ttsMs: undefined,
    });
  });

  it("generates web scoped ids", () => {
    expect(createSessionId()).toMatch(/^web-/);
    expect(createSpeakerId()).toMatch(/^web-speaker-/);
  });
});
