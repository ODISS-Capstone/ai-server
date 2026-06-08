import "@testing-library/jest-dom";
import { vi } from "vitest";

class MockWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  readyState = MockWebSocket.CONNECTING;
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;
  send = vi.fn();

  constructor(public url: string) {
    ((window as any).__mockWebSockets ||= []).push(this);
    if ((window as any).__mockWebSocketDisableAutoOpen) {
      return;
    }
    setTimeout(() => {
      this.readyState = MockWebSocket.OPEN;
      this.onopen?.();
    }, 0);
  }

  addEventListener(type: string, callback: () => void) {
    if (type === "open") {
      setTimeout(callback, 0);
    }
  }

  close() {
    this.readyState = MockWebSocket.CLOSED;
    this.onclose?.();
  }
}

Object.defineProperty(window, "WebSocket", {
  value: MockWebSocket,
  writable: true,
});

Object.defineProperty(window, "speechSynthesis", {
  value: {
    speak: vi.fn(),
    cancel: vi.fn(),
  },
  writable: true,
});

Object.defineProperty(window, "SpeechSynthesisUtterance", {
  value: class SpeechSynthesisUtterance {
    lang = "";
    rate = 1;
    onstart: (() => void) | null = null;
    onend: (() => void) | null = null;
    constructor(public text: string) {}
  },
  writable: true,
});
