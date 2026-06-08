export type LiveTextRole = "user" | "reply";

const LIVE_TEXT_LIMITS: Record<LiveTextRole, number> = {
  user: 72,
  reply: 118,
};

const LIVE_TEXT_ELLIPSIS = "\u2026";
const KOREAN_SENTENCE_ENDINGS = new Set(["\uC694", "\uB2E4"]);
const SENTENCE_PUNCTUATION = new Set([".", "!", "?", "\u3002", "\uFF01", "\uFF1F"]);
const SOFT_CUT_PUNCTUATION = new Set([...SENTENCE_PUNCTUATION, ",", "\uFF0C"]);

export function normalizeLiveText(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

export function compactLiveText(text: string, role: LiveTextRole): string {
  const normalized = normalizeLiveText(text);
  const maxLength = LIVE_TEXT_LIMITS[role];
  if (Array.from(normalized).length <= maxLength) {
    return normalized;
  }

  const firstSentence = firstDisplaySentence(normalized, maxLength);
  if (firstSentence && Array.from(firstSentence).length <= maxLength) {
    return appendLiveEllipsis(firstSentence);
  }

  const cut = safeLiveCutIndex(normalized, maxLength);
  const clipped = Array.from(normalized).slice(0, cut).join("").trim();
  return appendLiveEllipsis(clipped);
}

export function liveCaptionClass(text: string): string {
  const length = Array.from(normalizeLiveText(text)).length;
  if (length <= 18) return "caption-short";
  if (length <= 48) return "caption-medium";
  if (length <= 92) return "caption-long";
  return "caption-xlong";
}

function firstDisplaySentence(text: string, maxLength: number): string {
  const chars = Array.from(text);
  const minSentenceLength = Math.floor(maxLength * 0.35);
  for (let index = 0; index < chars.length; index += 1) {
    const char = chars[index];
    const next = chars[index + 1] || "";
    if (SENTENCE_PUNCTUATION.has(char)) {
      return chars.slice(0, index + 1).join("").trim();
    }
    if (index >= minSentenceLength && KOREAN_SENTENCE_ENDINGS.has(char) && (next === "" || next === " ")) {
      return chars.slice(0, index + 1).join("").trim();
    }
  }
  return "";
}

function safeLiveCutIndex(text: string, maxLength: number): number {
  const chars = Array.from(text);
  const start = Math.min(maxLength, chars.length - 1);
  const minSafeIndex = Math.floor(maxLength * 0.48);
  let lastSpace = -1;

  for (let index = start; index >= minSafeIndex; index -= 1) {
    const char = chars[index];
    const next = chars[index + 1] || "";
    if (SOFT_CUT_PUNCTUATION.has(char)) {
      return index + 1;
    }
    if (KOREAN_SENTENCE_ENDINGS.has(char) && (next === "" || next === " ")) {
      return index + 1;
    }
    if (char === " " && lastSpace === -1) {
      lastSpace = index;
    }
  }

  return lastSpace > minSafeIndex ? lastSpace : maxLength;
}

function appendLiveEllipsis(text: string): string {
  const trimmed = text.replace(/[,\s]+$/g, "").trim();
  return trimmed.endsWith("...") || trimmed.endsWith(LIVE_TEXT_ELLIPSIS) ? trimmed : `${trimmed}${LIVE_TEXT_ELLIPSIS}`;
}
