import { beforeEach, describe, expect, it, vi } from "vitest";

import { saveToken, sendTurnFeedback, storedToken, tokenFromUrl, websocketUrl } from "./assistant";
import { createMessage } from "../state/assistantSession";

describe("assistant api client", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
  });

  it("stores and clears the runtime tester token", () => {
    saveToken("  test-token  ");
    expect(storedToken()).toBe("test-token");

    saveToken("");
    expect(storedToken()).toBe("");
  });

  it("reads token from the URL and builds websocket URL", () => {
    window.history.replaceState({}, "", "/app?token=url-token");
    expect(tokenFromUrl()).toBe("url-token");

    const url = new URL(websocketUrl("ws-token"));
    expect(url.protocol).toMatch(/^ws/);
    expect(url.pathname).toBe("/ws/chat");
    expect(url.searchParams.get("token")).toBe("ws-token");
  });

  it("posts turn feedback with bearer token", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ success: true, stored_at: "now", path: "feedback.md" }),
    });
    vi.stubGlobal("fetch", fetchMock);

    const message = createMessage({
      sender: "odiss",
      text: "네, 말씀하세요.",
      responseType: "wake_word_ack",
      fastPath: "wake_word",
      latency: { firstMessageMs: 10, finalResponseMs: 12 },
    });

    await sendTurnFeedback({
      sessionId: "session-1",
      speakerId: "speaker-1",
      turnId: "turn-1",
      rating: "up",
      tags: ["wake_word_ack"],
      comment: "",
      userText: "오디스",
      message,
      token: "assistant-token",
    });

    const [path, init] = fetchMock.mock.calls[0];
    expect(path).toBe("/api/feedback/turn");
    expect(init.headers.Authorization).toBe("Bearer assistant-token");
    expect(JSON.parse(init.body).response_type).toBe("wake_word_ack");
  });
});
