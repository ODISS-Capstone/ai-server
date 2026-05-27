export type Sender = "user" | "odiss" | "filler" | "system" | "warning";

export interface LatencyMetrics {
  sttMs?: number;
  firstMessageMs?: number;
  finalResponseMs?: number;
  ttsMs?: number;
}

export interface AssistantMessage {
  id: string;
  turnId?: string;
  sender: Sender;
  text: string;
  createdAt: string;
  responseType?: string;
  fastPath?: string;
  stage?: string;
  reason?: string;
  requiresTts?: boolean;
  raw?: unknown;
  latency?: LatencyMetrics;
  feedback?: "up" | "down";
  userText?: string;
}

export interface TurnTiming {
  sttStart: number;
  sttEnd: number;
  wsSend: number;
  firstMessage: number;
  finalMessage: number;
  ttsStart: number;
  ttsEnd: number;
}

export type AssistantAction =
  | { type: "append"; message: AssistantMessage }
  | { type: "clear" }
  | { type: "feedback"; messageId: string; rating: "up" | "down" };

export function assistantMessagesReducer(
  messages: AssistantMessage[],
  action: AssistantAction,
): AssistantMessage[] {
  if (action.type === "clear") {
    return [];
  }
  if (action.type === "append") {
    return [action.message, ...messages].slice(0, 200);
  }
  return messages.map((message) =>
    message.id === action.messageId ? { ...message, feedback: action.rating } : message,
  );
}

export function createSessionId(): string {
  return `web-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

export function createSpeakerId(): string {
  return `web-speaker-${Math.random().toString(36).slice(2, 10)}`;
}

export function createTurnId(): string {
  return `turn-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

export function createMessage(input: Omit<AssistantMessage, "id" | "createdAt">): AssistantMessage {
  return {
    ...input,
    id: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`,
    createdAt: new Date().toISOString(),
  };
}

export function computeLatency(timing: Partial<TurnTiming>): LatencyMetrics {
  return {
    sttMs: elapsed(timing.sttStart, timing.sttEnd),
    firstMessageMs: elapsed(timing.wsSend, timing.firstMessage),
    finalResponseMs: elapsed(timing.wsSend, timing.finalMessage),
    ttsMs: elapsed(timing.ttsStart, timing.ttsEnd),
  };
}

function elapsed(start?: number, end?: number): number | undefined {
  if (!Number.isFinite(start) || !Number.isFinite(end) || !start || !end || end < start) {
    return undefined;
  }
  return Math.round(end - start);
}
