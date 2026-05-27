import type { AssistantMessage, LatencyMetrics } from "../state/assistantSession";

const TOKEN_STORAGE_KEY = "odiss.assistant.token";

export interface MedicationItem {
  name: string;
  strength?: string | null;
  dosage?: string | null;
  frequency?: string | null;
  timing?: string | null;
  raw_line?: string | null;
}

export interface OcrUploadResponse {
  raw_text: string;
  medications: MedicationItem[];
  success: boolean;
  message?: string | null;
}

export interface FeedbackResponse {
  success: boolean;
  stored_at: string;
  path: string;
}

export function apiBase(): string {
  return import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") ?? "";
}

export function storedToken(): string {
  return localStorage.getItem(TOKEN_STORAGE_KEY) ?? "";
}

export function saveToken(token: string): void {
  const normalized = token.trim();
  if (normalized) {
    localStorage.setItem(TOKEN_STORAGE_KEY, normalized);
  } else {
    localStorage.removeItem(TOKEN_STORAGE_KEY);
  }
}

export function tokenFromUrl(): string {
  const params = new URLSearchParams(window.location.search);
  return params.get("token")?.trim() ?? "";
}

export function websocketUrl(token: string): string {
  const base = apiBase() || window.location.origin;
  const url = new URL(base);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.pathname = "/ws/chat";
  url.search = "";
  if (token.trim()) {
    url.searchParams.set("token", token.trim());
  }
  return url.toString();
}

export async function uploadOcrImage(file: File, token: string): Promise<OcrUploadResponse> {
  const body = new FormData();
  body.set("file", file);
  const response = await fetch(`${apiBase()}/upload/image`, {
    method: "POST",
    headers: authHeaders(token),
    body,
  });
  return parseJson<OcrUploadResponse>(response);
}

export async function sendTurnFeedback(input: {
  sessionId: string;
  speakerId: string;
  turnId: string;
  rating: "up" | "down";
  tags: string[];
  comment: string;
  userText: string;
  message: AssistantMessage;
  token: string;
}): Promise<FeedbackResponse> {
  const response = await fetch(`${apiBase()}/api/feedback/turn`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(input.token) },
    body: JSON.stringify({
      session_id: input.sessionId,
      speaker_id: input.speakerId,
      turn_id: input.turnId,
      rating: input.rating,
      tags: input.tags,
      comment: input.comment,
      user_text: input.userText,
      response_text: input.message.text,
      response_type: input.message.responseType || "",
      fast_path: input.message.fastPath || "",
      latency: toFeedbackLatency(input.message.latency),
      raw: input.message.raw && typeof input.message.raw === "object" ? input.message.raw : {},
      user_agent: navigator.userAgent,
    }),
  });
  return parseJson<FeedbackResponse>(response);
}

export async function sendSessionFeedback(input: {
  sessionId: string;
  speakerId: string;
  satisfaction: number;
  comment: string;
  problemTags: string[];
  turnCount: number;
  token: string;
}): Promise<FeedbackResponse> {
  const response = await fetch(`${apiBase()}/api/feedback/session`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(input.token) },
    body: JSON.stringify({
      session_id: input.sessionId,
      speaker_id: input.speakerId,
      satisfaction: input.satisfaction,
      comment: input.comment,
      problem_tags: input.problemTags,
      turn_count: input.turnCount,
      user_agent: navigator.userAgent,
    }),
  });
  return parseJson<FeedbackResponse>(response);
}

function authHeaders(token: string): HeadersInit {
  return token.trim() ? { Authorization: `Bearer ${token.trim()}` } : {};
}

function toFeedbackLatency(latency?: LatencyMetrics): Record<string, number | undefined> {
  return {
    stt_ms: latency?.sttMs,
    first_message_ms: latency?.firstMessageMs,
    final_response_ms: latency?.finalResponseMs,
    tts_ms: latency?.ttsMs,
  };
}

async function parseJson<T>(response: Response): Promise<T> {
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = payload?.detail || payload?.message || `Request failed: ${response.status}`;
    throw new Error(String(detail));
  }
  return payload as T;
}
