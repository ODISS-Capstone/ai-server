export type ClientEventType =
  | "camera_close"
  | "camera_error"
  | "camera_open"
  | "camera_ready"
  | "ocr_error"
  | "ocr_start"
  | "ocr_success"
  | "stt_error"
  | "stt_result"
  | "stt_start"
  | "tts_end"
  | "tts_start"
  | "tts_stop"
  | "ws_close"
  | "ws_connecting"
  | "ws_error"
  | "ws_open"
  | "ws_reconnect_visible";

export interface ClientEvent {
  id: string;
  at: string;
  type: ClientEventType;
  detail?: Record<string, unknown>;
}

const MAX_CLIENT_EVENTS = 120;

export function createClientEvent(type: ClientEventType, detail?: Record<string, unknown>): ClientEvent {
  return {
    at: new Date().toISOString(),
    detail,
    id: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`,
    type,
  };
}

export function appendClientEvent(events: ClientEvent[], event: ClientEvent): ClientEvent[] {
  return [...events, event].slice(-MAX_CLIENT_EVENTS);
}
