import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import App from "../App";

vi.mock("../api/memoryBrowser", () => ({
  getPatientDetail: vi.fn(),
  getPatientRecords: vi.fn(),
  readMemoryEntry: vi.fn(),
  searchPatients: vi.fn().mockResolvedValue({
    patients: [],
    query: "",
    total: 0,
  }),
}));

describe("Assistant public live text", () => {
  beforeEach(() => {
    window.history.pushState({}, "", "/");
    (window as any).__mockWebSockets = [];
    (window as any).__mockWebSocketDisableAutoOpen = false;
    vi.clearAllMocks();
    vi.useRealTimers();
  });

  it("compacts a long public assistant reply and preserves the full text as title", async () => {
    render(<App />);
    const socket = await latestSocket();
    const longReply =
      "Please check the medication record first before taking the same medicine again. This second sentence should stay available in the full debug log.";

    act(() => {
      socket.onmessage?.({
        data: JSON.stringify({
          requires_tts: true,
          response_text: longReply,
          response_type: "medical_response",
          type: "response",
        }),
      });
    });

    const card = await screen.findByTitle(longReply);
    expect(card).toHaveClass("is-compact");
    expect(card).toHaveTextContent("ODISS");
    expect(card).toHaveTextContent("Please check the medication record first");
    expect(card.textContent).toContain("\u2026");
    expect(card).not.toHaveTextContent("This second sentence");
  });

  it("keeps the full assistant reply visible in admin logs", async () => {
    window.history.pushState({}, "", "/?admin=1");
    render(<App />);
    const socket = await latestSocket();
    const longReply =
      "Please check the medication record first before taking the same medicine again. This full response should remain visible for admin review.";

    act(() => {
      socket.onmessage?.({
        data: JSON.stringify({
          requires_tts: true,
          response_text: longReply,
          response_type: "medical_response",
          type: "response",
        }),
      });
    });

    await waitFor(() => {
      expect(screen.getAllByText(longReply).length).toBeGreaterThanOrEqual(1);
    });
  });

  it("replays the latest ODISS reply without sending a server turn", async () => {
    render(<App />);
    const socket = await latestSocket();
    const reply = "네, 김영수님. 말씀하세요.";

    act(() => {
      socket.onmessage?.({
        data: JSON.stringify({
          requires_tts: true,
          response_text: reply,
          response_type: "wake_word_ack",
          type: "response",
        }),
      });
    });

    const speakMock = vi.mocked(window.speechSynthesis.speak);
    await waitFor(() => expect(speakMock).toHaveBeenCalledTimes(1));
    speakMock.mockClear();

    fireEvent.click(screen.getByRole("button", { name: "다시 듣기" }));

    expect(speakMock).toHaveBeenCalledTimes(1);
    expect(socket.send).not.toHaveBeenCalledWith(expect.stringContaining(reply));
  });

  it("shows a stop voice control while browser TTS is speaking", async () => {
    render(<App />);
    const socket = await latestSocket();

    act(() => {
      socket.onmessage?.({
        data: JSON.stringify({
          requires_tts: true,
          response_text: "다시 들려드릴게요.",
          response_type: "smalltalk",
          type: "response",
        }),
      });
    });

    const speakMock = vi.mocked(window.speechSynthesis.speak);
    await waitFor(() => expect(speakMock).toHaveBeenCalledTimes(1));
    const utterance = speakMock.mock.calls[0][0] as SpeechSynthesisUtterance;
    act(() => {
      utterance.onstart?.(new Event("start") as SpeechSynthesisEvent);
    });

    fireEvent.click(screen.getByRole("button", { name: "음성 중지" }));

    expect(window.speechSynthesis.cancel).toHaveBeenCalledTimes(1);
  });

  it("opens manual input with a short notice when browser STT is unsupported", async () => {
    render(<App />);
    await latestSocket();

    fireEvent.click(screen.getByRole("button", { name: "직접 입력하기" }));

    expect(screen.getByText("음성 입력이 어려워 직접 입력을 열었어요.")).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "대화 입력" })).toBeInTheDocument();
  });

  it("shows a reconnect notice when the WebSocket stays closed", async () => {
    render(<App />);
    const socket = await latestSocket();

    await waitFor(() => {
      expect(socket.send).toHaveBeenCalledWith(JSON.stringify({ type: "ping" }));
    });

    vi.useFakeTimers();
    (window as any).__mockWebSocketDisableAutoOpen = true;
    act(() => {
      socket.close();
    });
    act(() => {
      vi.advanceTimersByTime(3000);
    });

    expect(screen.getByText("연결 다시 시도 중이에요.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "다시 연결" })).toBeInTheDocument();
  });

  it("keeps client events visible only in admin debug", async () => {
    window.history.pushState({}, "", "/?admin=1");
    render(<App />);
    const socket = await latestSocket();

    act(() => {
      socket.onmessage?.({
        data: JSON.stringify({
          requires_tts: true,
          response_text: "관리자 로그 확인입니다.",
          response_type: "smalltalk",
          type: "response",
        }),
      });
    });

    await waitFor(() => {
      expect(screen.getByText(/클라이언트 이벤트/)).toBeInTheDocument();
    });
  });
});

async function latestSocket(): Promise<any> {
  await waitFor(() => {
    expect((window as any).__mockWebSockets.length).toBeGreaterThan(0);
  });
  const sockets = (window as any).__mockWebSockets;
  return sockets[sockets.length - 1];
}
