export interface BrowserCapabilities {
  mediaDevices: boolean;
  mediaRecorder: boolean;
  speechRecognition: boolean;
  speechSynthesis: boolean;
}

export function detectBrowserCapabilities(
  win: Window | undefined = typeof window === "undefined" ? undefined : window,
  nav: Navigator | undefined = typeof navigator === "undefined" ? undefined : navigator,
): BrowserCapabilities {
  return {
    mediaDevices: Boolean(nav?.mediaDevices?.getUserMedia),
    mediaRecorder: Boolean(win && "MediaRecorder" in win && nav?.mediaDevices?.getUserMedia),
    speechRecognition: Boolean(speechRecognitionConstructor(win)),
    speechSynthesis: Boolean(win && "speechSynthesis" in win),
  };
}

export function speechRecognitionConstructor(win: Window | undefined = typeof window === "undefined" ? undefined : window) {
  return win ? ((win as any).SpeechRecognition || (win as any).webkitSpeechRecognition) : null;
}
