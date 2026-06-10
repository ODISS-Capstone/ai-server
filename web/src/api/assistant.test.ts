import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  resolveApiBase,
  saveToken,
  sendTurnFeedback,
  storedToken,
  tokenFromUrl,
  transcribeAudio,
  websocketUrl,
} from "./assistant";
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

  it("ignores localhost API base on deployed origins", () => {
    expect(resolveApiBase("http://127.0.0.1:8000", "https://www.odiss.p-e.kr")).toBe("");
    expect(resolveApiBase("http://127.0.0.1:8000", "http://localhost:5173")).toBe("http://127.0.0.1:8000");
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
    expect(new URL(String(path), window.location.origin).pathname).toBe("/api/feedback/turn");
    expect(init.headers.Authorization).toBe("Bearer assistant-token");
    expect(JSON.parse(init.body).response_type).toBe("wake_word_ack");
  });

  it("uploads audio for server STT with speaker metadata", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({
        audio_bytes: 12,
        model: "gemini-2.5-flash",
        provider: "gemini",
        success: true,
        text: "오디스 약 알려줘",
      }),
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await transcribeAudio(
      new File([new Blob(["audio"])], "voice.webm", { type: "audio/webm" }),
      { speakerId: "speaker-1", token: "assistant-token", language: "ko-KR" },
    );

    const [path, init] = fetchMock.mock.calls[0];
    expect(new URL(String(path), window.location.origin).pathname).toBe("/api/stt/transcribe");
    expect(init.method).toBe("POST");
    expect(init.headers.Authorization).toBe("Bearer assistant-token");
    expect(init.body).toBeInstanceOf(FormData);
    expect((init.body as FormData).get("speaker_id")).toBe("speaker-1");
    expect((init.body as FormData).get("language")).toBe("ko-KR");
    expect((init.body as FormData).get("file")).toBeInstanceOf(File);
    expect(result.text).toBe("오디스 약 알려줘");
  });
});
