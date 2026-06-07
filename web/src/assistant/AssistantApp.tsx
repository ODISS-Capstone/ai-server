import { useEffect, useMemo, useReducer, useRef, useState } from "react";
import type { CSSProperties, ChangeEvent, FormEvent } from "react";

import { uploadOcrImage, websocketUrl } from "../api/assistant";
import {
  assistantMessagesReducer,
  computeLatency,
  createMessage,
  createSessionId,
  createSpeakerId,
  createTurnId,
} from "../state/assistantSession";
import type { AssistantMessage, TurnTiming } from "../state/assistantSession";

type ConnectionStatus = "connecting" | "connected" | "closed" | "error";
type CameraMode = "idle" | "opening" | "ready" | "captured" | "error";
type AssistantMode = "idle" | "listening" | "thinking" | "camera_ready" | "ocr_processing" | "speaking" | "error";

interface AssistantAppProps {
  token: string;
  adminMode: boolean;
}

interface WsPayload {
  type?: string;
  text?: string;
  message?: string;
  response_text?: string;
  response_type?: string;
  fast_path?: string;
  stage?: string;
  reason?: string;
  requires_tts?: boolean;
  server_elapsed_ms?: number;
  ws_elapsed_ms?: number;
  turn_id?: string;
  session_id?: string;
  needs_recapture?: boolean;
  [key: string]: unknown;
}

const SPEAKER_STORAGE_KEY = "odiss.assistant.speaker_id";
const SESSION_STORAGE_KEY = "odiss.assistant.session_id";

export default function AssistantApp({ token, adminMode }: AssistantAppProps) {
  const [messages, dispatch] = useReducer(assistantMessagesReducer, []);
  const [manualText, setManualText] = useState("");
  const [interimText, setInterimText] = useState("");
  const [status, setStatus] = useState<ConnectionStatus>("closed");
  const [ttsEnabled, setTtsEnabled] = useState(true);
  const [fillerTtsEnabled, setFillerTtsEnabled] = useState(true);
  const [voiceArmed, setVoiceArmed] = useState(false);
  const [listening, setListening] = useState(false);
  const [voiceLevel, setVoiceLevel] = useState(0);
  const [sttPulse, setSttPulse] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [manualOpen, setManualOpen] = useState(false);
  const [cameraMode, setCameraMode] = useState<CameraMode>("idle");
  const [cameraMessage, setCameraMessage] = useState("약봉투나 처방전을 보여주시면 제가 읽고 대화로 이어갈게요.");
  const [ocrPreview, setOcrPreview] = useState("");
  const [ocrBusy, setOcrBusy] = useState(false);
  const [speakerId, setSpeakerId] = useState(
    () => localStorage.getItem(SPEAKER_STORAGE_KEY) || createSpeakerId(),
  );
  const [sessionId, setSessionId] = useState(
    () => localStorage.getItem(SESSION_STORAGE_KEY) || createSessionId(),
  );

  const wsRef = useRef<WebSocket | null>(null);
  const recognitionRef = useRef<any>(null);
  const voiceArmedRef = useRef(false);
  const manualStopRef = useRef(false);
  const sttPulseTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const micContextRef = useRef<AudioContext | null>(null);
  const micSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const micAnalyserRef = useRef<AnalyserNode | null>(null);
  const micMeterFrameRef = useRef<number | null>(null);
  const speakingRef = useRef(false);
  const activeSpeechTextRef = useRef("");
  const lastInterruptRef = useRef<{ at: number; text: string } | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const turnTimingRef = useRef<Partial<TurnTiming> & { userText?: string }>({});
  const turnTimingsRef = useRef<Map<string, Partial<TurnTiming> & { userText?: string }>>(new Map());
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const ocrRunRef = useRef(0);

  const speechSupported = useMemo(() => {
    if (typeof window === "undefined") {
      return false;
    }
    return Boolean((window as any).SpeechRecognition || (window as any).webkitSpeechRecognition);
  }, []);

  useEffect(() => {
    localStorage.setItem(SPEAKER_STORAGE_KEY, speakerId);
  }, [speakerId]);

  useEffect(() => {
    localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
  }, [sessionId]);

  useEffect(() => {
    voiceArmedRef.current = voiceArmed;
  }, [voiceArmed]);

  useEffect(() => {
    if (voiceArmed && speechSupported) {
      void startVoiceMeter();
    } else {
      stopVoiceMeter();
    }
    // The meter follows the user-controlled armed state only.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [speechSupported, voiceArmed]);

  useEffect(() => {
    return () => {
      if (sttPulseTimerRef.current) {
        clearTimeout(sttPulseTimerRef.current);
      }
      stopVoiceMeter();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      const socket = wsRef.current;
      if (socket) {
        socket.onclose = null;
        socket.close();
      }
      wsRef.current = null;
      stopVoiceMeter();
      stopCamera();
    };
    // Reconnect only when the credential or speaker changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token, speakerId]);

  function connectWebSocket(): WebSocket | null {
    if (typeof WebSocket === "undefined") {
      setStatus("error");
      return null;
    }
    const existing = wsRef.current;
    if (
      existing &&
      (existing.readyState === WebSocket.OPEN || existing.readyState === WebSocket.CONNECTING)
    ) {
      return existing;
    }

    setStatus("connecting");
    const socket = new WebSocket(websocketUrl(token));
    wsRef.current = socket;

    socket.onopen = () => {
      setStatus("connected");
      socket.send(JSON.stringify({ type: "ping" }));
    };
    socket.onmessage = (event) => handleSocketMessage(event.data);
    socket.onerror = () => setStatus("error");
    socket.onclose = () => {
      setStatus("closed");
      if (!reconnectTimerRef.current) {
        reconnectTimerRef.current = setTimeout(() => {
          reconnectTimerRef.current = null;
          connectWebSocket();
        }, 900);
      }
    };
    return socket;
  }

  function sendPayload(payload: Record<string, unknown>) {
    const socket = connectWebSocket();
    if (!socket) {
      appendSystemMessage("브라우저에서 WebSocket을 사용할 수 없습니다.", "warning");
      return;
    }
    const serialized = JSON.stringify(payload);
    if (socket.readyState === WebSocket.OPEN) {
      socket.send(serialized);
      return;
    }
    socket.addEventListener("open", () => socket.send(serialized), { once: true });
  }

  function handleSocketMessage(rawData: string) {
    let payload: WsPayload;
    try {
      payload = JSON.parse(rawData) as WsPayload;
    } catch {
      appendSystemMessage("서버 메시지를 읽지 못했습니다.", "warning");
      return;
    }

    if (payload.type === "pong") {
      return;
    }

    if (payload.type === "ocr_request" || payload.needs_recapture) {
      void activateCamera("서버가 사진 확인을 요청했습니다. 약봉투나 처방전을 화면에 맞춰주세요.");
    }

    const text = payloadText(payload);
    if (payloadRequestsCameraClose(payload, text)) {
      closeCameraSession();
    }

    const turnId = typeof payload.turn_id === "string" ? payload.turn_id : undefined;
    const timing = (turnId && turnTimingsRef.current.get(turnId)) || turnTimingRef.current;
    const now = performance.now();
    if (!timing.firstMessage) {
      timing.firstMessage = now;
    }
    if (isFinalPayload(payload)) {
      timing.finalMessage = now;
    }
    if (turnId) {
      turnTimingsRef.current.set(turnId, timing);
    }

    const sender = payloadSender(payload);
    const message = createMessage({
      turnId,
      sender,
      text,
      responseType: String(payload.response_type || payload.type || ""),
      fastPath: typeof payload.fast_path === "string" ? payload.fast_path : undefined,
      stage: typeof payload.stage === "string" ? payload.stage : undefined,
      reason: typeof payload.reason === "string" ? payload.reason : undefined,
      requiresTts: payload.requires_tts !== false,
      raw: payload,
      latency: computeLatency(timing),
      userText: timing.userText || "",
    });
    dispatch({ type: "append", message });

    if (payload.type === "session_closed") {
      wsRef.current?.close();
      connectWebSocket();
    }

    if (message.requiresTts && text) {
      const shouldSpeak = sender === "odiss" || (sender === "filler" && fillerTtsEnabled);
      if (shouldSpeak) {
        speak(text, turnId);
      }
    }
  }

  function sendText(text: string, source: "manual" | "speech" = "manual") {
    const normalized = text.trim();
    if (!normalized) {
      return;
    }
    const turnId = createTurnId();
    const now = performance.now();
    const previousTiming = turnTimingRef.current;
    const hasCurrentSttTiming = !previousTiming.wsSend;
    const timing = {
      sttStart: hasCurrentSttTiming ? previousTiming.sttStart : undefined,
      sttEnd: hasCurrentSttTiming ? previousTiming.sttEnd || now : now,
      wsSend: now,
      userText: normalized,
    };
    turnTimingRef.current = timing;
    turnTimingsRef.current.set(turnId, timing);
    dispatch({
      type: "append",
      message: createMessage({
        turnId,
        sender: "user",
        text: normalized,
        raw: { speaker_id: speakerId, session_id: sessionId, turn_id: turnId },
      }),
    });
    const shouldCloseCamera = isCameraDismissIntent(normalized) &&
      (showCameraPanel(cameraMode, ocrPreview, ocrBusy) || mentionsCameraSurface(normalized));
    if (shouldCloseCamera) {
      closeCameraSession();
    } else if (isPhotoIntent(normalized)) {
      void activateCamera("사진 확인을 준비했습니다. 약 이름이 잘 보이게 약봉투나 처방전을 보여주세요.");
    }
    sendPayload({
      type: "stt_result",
      text: normalized,
      speaker_id: speakerId,
      session_id: sessionId,
      turn_id: turnId,
      client_sent_at: new Date().toISOString(),
      client_context: {
        source,
        voice_armed: voiceArmedRef.current,
        listening,
        speaking: speakingRef.current,
        interrupted_tts:
          Boolean(lastInterruptRef.current) &&
          performance.now() - (lastInterruptRef.current?.at ?? 0) < 2500,
        interrupted_tts_preview: lastInterruptRef.current?.text.slice(0, 160) ?? "",
        camera_mode: cameraMode,
        tts_enabled: ttsEnabled,
        filler_tts_enabled: fillerTtsEnabled,
        interim_text: interimText,
        user_agent: navigator.userAgent,
        language: navigator.language,
      },
    });
    lastInterruptRef.current = null;
    setManualText("");
    setInterimText("");
  }

  function handleManualSubmit(event: FormEvent) {
    event.preventDefault();
    sendText(manualText);
  }

  function startAssistant() {
    if (!speechSupported) {
      setManualOpen(true);
      appendSystemMessage("이 브라우저는 음성 인식을 지원하지 않습니다. 아래에 말씀을 적어 주세요.", "warning");
      return;
    }
    setVoiceArmed(true);
    voiceArmedRef.current = true;
    void startVoiceMeter();
    startListening();
  }

  function pulseFromTranscript() {
    setSttPulse(true);
    if (sttPulseTimerRef.current) {
      clearTimeout(sttPulseTimerRef.current);
    }
    sttPulseTimerRef.current = setTimeout(() => {
      setSttPulse(false);
      sttPulseTimerRef.current = null;
    }, 850);
  }

  async function startVoiceMeter() {
    if (micStreamRef.current || !navigator.mediaDevices?.getUserMedia) {
      return;
    }
    const AudioContextConstructor = window.AudioContext || (window as any).webkitAudioContext;
    if (!AudioContextConstructor) {
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
        },
        video: false,
      });
      const context = new AudioContextConstructor();
      const analyser = context.createAnalyser();
      analyser.fftSize = 512;
      analyser.smoothingTimeConstant = 0.7;
      const source = context.createMediaStreamSource(stream);
      source.connect(analyser);
      micStreamRef.current = stream;
      micContextRef.current = context;
      micSourceRef.current = source;
      micAnalyserRef.current = analyser;

      const data = new Uint8Array(analyser.fftSize);
      const tick = () => {
        const currentAnalyser = micAnalyserRef.current;
        if (!currentAnalyser) {
          return;
        }
        currentAnalyser.getByteTimeDomainData(data);
        let sum = 0;
        for (let index = 0; index < data.length; index += 1) {
          const centered = (data[index] - 128) / 128;
          sum += centered * centered;
        }
        const rms = Math.sqrt(sum / data.length);
        const normalized = Math.min(1, Math.max(0, (rms - 0.016) * 10));
        setVoiceLevel((previous) => (Math.abs(previous - normalized) > 0.025 ? normalized : previous));
        micMeterFrameRef.current = requestAnimationFrame(tick);
      };
      tick();
    } catch {
      setVoiceLevel(0);
    }
  }

  function stopVoiceMeter() {
    if (micMeterFrameRef.current !== null) {
      cancelAnimationFrame(micMeterFrameRef.current);
      micMeterFrameRef.current = null;
    }
    micSourceRef.current?.disconnect();
    micSourceRef.current = null;
    micAnalyserRef.current = null;
    micStreamRef.current?.getTracks().forEach((track) => track.stop());
    micStreamRef.current = null;
    void micContextRef.current?.close();
    micContextRef.current = null;
    setVoiceLevel(0);
  }

  function startListening() {
    const Recognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!Recognition) {
      appendSystemMessage("이 브라우저는 음성 인식을 지원하지 않습니다. 텍스트로 입력해 주세요.", "warning");
      return;
    }
    if (listening || recognitionRef.current) {
      return;
    }
    manualStopRef.current = false;
    const recognition = new Recognition();
    recognitionRef.current = recognition;
    recognition.lang = "ko-KR";
    recognition.interimResults = true;
    recognition.continuous = false;
    recognition.maxAlternatives = 1;
    turnTimingRef.current = { sttStart: performance.now() };

    recognition.onstart = () => setListening(true);
    recognition.onerror = (event: { error?: string }) => {
      setListening(false);
      recognitionRef.current = null;
      const recoverable = ["no-speech", "aborted", "network"].includes(String(event.error || ""));
      if (!recoverable) {
        voiceArmedRef.current = false;
        setVoiceArmed(false);
        appendSystemMessage(`음성 인식 오류: ${event.error || "unknown"}`, "warning");
      }
    };
    recognition.onend = () => {
      setListening(false);
      recognitionRef.current = null;
      if (voiceArmedRef.current && !manualStopRef.current) {
        setTimeout(() => startListening(), 300);
      }
    };
    recognition.onresult = (event: any) => {
      let finalTranscript = "";
      let interimTranscript = "";
      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const result = event.results[index];
        const transcript = String(result[0]?.transcript || "");
        if (result.isFinal) {
          finalTranscript += transcript;
        } else {
          interimTranscript += transcript;
        }
      }
      const heardText = (finalTranscript || interimTranscript).trim();
      setInterimText(interimTranscript.trim());
      if (heardText) {
        pulseFromTranscript();
      }
      if (
        speakingRef.current &&
        heardText &&
        !isLikelyAssistantEcho(heardText, activeSpeechTextRef.current)
      ) {
        lastInterruptRef.current = {
          at: performance.now(),
          text: activeSpeechTextRef.current,
        };
        stopTts();
      }
      if (
        finalTranscript.trim() &&
        !(speakingRef.current && isLikelyAssistantEcho(finalTranscript, activeSpeechTextRef.current))
      ) {
        turnTimingRef.current.sttEnd = performance.now();
        sendText(finalTranscript, "speech");
      }
    };
    recognition.start();
  }

  function stopListening() {
    manualStopRef.current = true;
    voiceArmedRef.current = false;
    setVoiceArmed(false);
    recognitionRef.current?.stop?.();
    recognitionRef.current = null;
    setListening(false);
    stopVoiceMeter();
  }

  function speak(text: string, turnId?: string) {
    if (!ttsEnabled || typeof window === "undefined" || !("speechSynthesis" in window)) {
      return;
    }
    const timing = turnId ? turnTimingsRef.current.get(turnId) : undefined;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "ko-KR";
    utterance.rate = 1.02;
    activeSpeechTextRef.current = text;
    utterance.onstart = () => {
      speakingRef.current = true;
      setSpeaking(true);
      const now = performance.now();
      turnTimingRef.current.ttsStart = now;
      if (timing) {
        timing.ttsStart = now;
        turnTimingsRef.current.set(turnId!, timing);
      }
      if (voiceArmedRef.current && !recognitionRef.current && !manualStopRef.current) {
        setTimeout(() => startListening(), 120);
      }
    };
    utterance.onend = () => {
      speakingRef.current = false;
      activeSpeechTextRef.current = "";
      setSpeaking(false);
      const now = performance.now();
      turnTimingRef.current.ttsEnd = now;
      if (timing) {
        timing.ttsEnd = now;
        turnTimingsRef.current.set(turnId!, timing);
      }
      if (voiceArmedRef.current && !manualStopRef.current) {
        setTimeout(() => startListening(), 250);
      }
    };
    utterance.onerror = () => {
      speakingRef.current = false;
      activeSpeechTextRef.current = "";
      setSpeaking(false);
      if (voiceArmedRef.current && !manualStopRef.current) {
        setTimeout(() => startListening(), 250);
      }
    };
    window.speechSynthesis.speak(utterance);
  }

  function stopTts() {
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
    speakingRef.current = false;
    activeSpeechTextRef.current = "";
    setSpeaking(false);
  }

  async function activateCamera(message = "사진 확인을 준비했습니다.") {
    setCameraMessage(message);
    setCameraMode("opening");
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraMode("error");
      setCameraMessage("이 브라우저에서는 실시간 카메라를 열 수 없습니다. 사진 업로드를 사용해 주세요.");
      return;
    }
    try {
      stopCamera();
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => undefined);
      }
      setCameraMode("ready");
      setCameraMessage("카메라가 준비됐습니다. 약 이름이 크게 보이게 맞춘 뒤 촬영해 주세요.");
    } catch {
      setCameraMode("error");
      setCameraMessage("카메라 권한을 받지 못했습니다. 파일 업로드로도 OCR 확인이 가능합니다.");
    }
  }

  function stopCamera() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }

  function closeCameraSession(message = "") {
    ocrRunRef.current += 1;
    setOcrBusy(false);
    stopCamera();
    setCameraMode("idle");
    updatePreview(null);
    setCameraMessage("약봉투나 처방전을 보여주시면 제가 읽고 대화로 이어갈게요.");
    if (message) {
      appendSystemMessage(message);
    }
  }

  async function captureAndAnalyze() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.videoWidth === 0) {
      appendSystemMessage("카메라 화면이 아직 준비되지 않았습니다.", "warning");
      return;
    }
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d")?.drawImage(video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
    if (!blob) {
      appendSystemMessage("사진을 만들지 못했습니다. 다시 시도해 주세요.", "warning");
      return;
    }
    const file = new File([blob], `odiss-ocr-${Date.now()}.jpg`, { type: "image/jpeg" });
    updatePreview(file);
    setCameraMode("captured");
    stopCamera();
    await analyzeOcrFile(file);
  }

  function handleOcrFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] || null;
    updatePreview(file);
    if (file) {
      setCameraMode("captured");
      setCameraMessage("사진이 준비됐습니다. 바로 확인하겠습니다.");
      void analyzeOcrFile(file);
    }
  }

  function updatePreview(file: File | null) {
    if (ocrPreview) {
      URL.revokeObjectURL(ocrPreview);
    }
    setOcrPreview(file ? URL.createObjectURL(file) : "");
  }

  async function analyzeOcrFile(file: File) {
    const runId = ocrRunRef.current + 1;
    ocrRunRef.current = runId;
    setOcrBusy(true);
    appendSystemMessage("사진을 읽고 있습니다. 글자가 잘 보이는지 확인한 뒤 대화로 이어갈게요.");
    try {
      const result = await uploadOcrImage(file, token);
      if (ocrRunRef.current !== runId) {
        return;
      }
      appendSystemMessage(
        result.medications.length
          ? `OCR에서 약 ${result.medications.length}개를 찾았습니다. 저장 여부를 대화로 확인할게요.`
          : "약 이름이 선명하지 않습니다. 필요하면 다시 촬영을 안내할게요.",
      );
      const turnId = createTurnId();
      const timing = {
        wsSend: performance.now(),
        userText: "OCR image upload",
      };
      turnTimingRef.current = timing;
      turnTimingsRef.current.set(turnId, timing);
      sendPayload({
        type: "ocr_result",
        speaker_id: speakerId,
        session_id: sessionId,
        turn_id: turnId,
        client_sent_at: new Date().toISOString(),
        client_context: {
          source: "ocr",
          camera_mode: cameraMode,
          ocr_file_type: file.type,
          ocr_file_size: file.size,
          user_agent: navigator.userAgent,
          language: navigator.language,
        },
        data: result,
      });
    } catch (error) {
      if (ocrRunRef.current !== runId) {
        return;
      }
      appendSystemMessage(error instanceof Error ? error.message : "OCR 업로드에 실패했습니다.", "warning");
    } finally {
      if (ocrRunRef.current === runId) {
        setOcrBusy(false);
      }
    }
  }

  function startNewUser() {
    const nextSpeakerId = createSpeakerId();
    const nextSessionId = createSessionId();
    setSpeakerId(nextSpeakerId);
    setSessionId(nextSessionId);
    dispatch({ type: "clear" });
    closeCameraSession();
    appendSystemMessage("새 사용자 세션을 시작했습니다. 처음 사용자처럼 테스트할 수 있습니다.");
  }

  function appendSystemMessage(text: string, sender: "system" | "warning" = "system") {
    dispatch({ type: "append", message: createMessage({ sender, text }) });
  }

  function exportSessionLog() {
    const exportedAt = new Date().toISOString();
    const log = {
      version: 1,
      exported_at: exportedAt,
      session: {
        session_id: sessionId,
        speaker_id: speakerId,
        admin_mode: adminMode,
        token_configured: Boolean(token),
        location: window.location.href,
      },
      runtime: {
        status,
        websocket_ready_state: wsRef.current?.readyState ?? null,
        speech_supported: speechSupported,
        speech_synthesis_supported: typeof window !== "undefined" && "speechSynthesis" in window,
        media_devices_supported: Boolean(navigator.mediaDevices?.getUserMedia),
        tts_enabled: ttsEnabled,
        filler_tts_enabled: fillerTtsEnabled,
        voice_armed: voiceArmed,
        listening,
        voice_level: Number(voiceLevel.toFixed(2)),
        voice_detected: voiceLevel > 0.12 || sttPulse || Boolean(interimText.trim()),
        speaking,
        interim_text: interimText,
        camera_mode: cameraMode,
        camera_message: cameraMessage,
        ocr_busy: ocrBusy,
        has_ocr_preview: Boolean(ocrPreview),
        user_agent: navigator.userAgent,
        language: navigator.language,
        pending_turn_count: turnTimingsRef.current.size,
      },
      messages: [...messages].reverse().map((message) => ({
        id: message.id,
        turn_id: message.turnId,
        created_at: message.createdAt,
        sender: message.sender,
        text: message.text,
        response_type: message.responseType,
        fast_path: message.fastPath,
        stage: message.stage,
        reason: message.reason,
        requires_tts: message.requiresTts,
        latency: message.latency,
        user_text: message.userText,
        raw: message.raw,
      })),
    };
    downloadJson(`odiss-session-${sessionId}-${compactTimestamp(exportedAt)}.json`, log);
  }

  const chronologicalMessages = [...messages].reverse();
  const latestMessage = messages[0];
  const latestUserMessage = messages.find((message) => message.sender === "user");
  const latestReplyMessage = messages.find((message) => isLiveReplyMessage(message, adminMode));
  const liveUserText = interimText.trim() || latestUserMessage?.text || "";
  const liveReplyText = latestReplyMessage?.text || "";
  const liveReplySender: AssistantMessage["sender"] = latestReplyMessage?.sender || "odiss";
  const hasLiveDialog = Boolean(liveUserText || liveReplyText);
  const liveUserSize = captionSizeClass(liveUserText);
  const liveReplySize = captionSizeClass(liveReplyText);
  const mode = currentAssistantMode({
    cameraMode,
    latestMessage,
    listening,
    ocrBusy,
    speaking,
    status,
    voiceArmed,
  });
  const stateCopy = assistantModeCopy(mode, {
    cameraMessage,
    interimText,
    speechSupported,
    status,
    voiceArmed,
  });
  const showCamera = showCameraPanel(cameraMode, ocrPreview, ocrBusy);
  const primaryLabel = primaryActionLabel({
    cameraMode,
    listening,
    ocrBusy,
    speechSupported,
    voiceArmed,
  });
  const primaryKind = primaryActionKind({ cameraMode, ocrBusy });
  const voiceDetected = primaryKind === "mic" && (voiceLevel > 0.12 || sttPulse || Boolean(interimText.trim()));
  const primaryButtonStyle = {
    "--voice-level": Math.max(voiceLevel, voiceDetected ? 0.36 : 0).toFixed(2),
  } as CSSProperties;

  function handlePrimaryAction() {
    if (ocrBusy || cameraMode === "opening") {
      return;
    }
    if (cameraMode === "ready") {
      void captureAndAnalyze();
      return;
    }
    if (cameraMode === "error") {
      fileInputRef.current?.click();
      return;
    }
    if (listening) {
      stopListening();
      return;
    }
    if (voiceArmed) {
      stopListening();
      return;
    }
    startAssistant();
  }

  const showVisibleCopy = adminMode || showCamera || mode === "error";

  return (
    <section className={`assistant-stage ${mode} ${adminMode ? "admin-view" : "public-view"} ${hasLiveDialog ? "has-live-dialog" : ""}`}>
      {adminMode ? (
        <div className="assistant-status-row">
          <span className={`status-dot ${status}`} />
          <span>{statusLabel(status)}</span>
        </div>
      ) : null}

      <main className={`assistant-center ${showCamera ? "camera-layout" : "voice-layout"}`} aria-label="ODISS 음성 비서">
        {showVisibleCopy ? (
          <div className="assistant-copy">
            <p className="assistant-kicker">{stateCopy.kicker}</p>
            <h1>{stateCopy.title}</h1>
            <p className="assistant-subtitle">{stateCopy.subtitle}</p>
          </div>
        ) : null}

        {showCamera ? (
          <section className="camera-focus" aria-label="약봉투 사진 확인">
            <div className="camera-stage">
              {cameraMode === "ready" || cameraMode === "opening" ? (
                <video ref={videoRef} className="camera-video" playsInline muted />
              ) : ocrPreview ? (
                <img className="ocr-preview" src={ocrPreview} alt="업로드 미리보기" />
              ) : (
                <div className="camera-placeholder">
                  <strong>약봉투를 화면에 맞춰주세요</strong>
                  <span>약 이름이 보이면 제가 사진을 읽겠습니다.</span>
                </div>
              )}
              <canvas ref={canvasRef} hidden />
            </div>
          </section>
        ) : null}

        {!adminMode && liveUserText ? (
          <section className="live-dialog above-mic" aria-label="내 말" aria-live="polite">
            <article className={`live-card user ${liveUserSize}`}>
              <p>{liveUserText}</p>
            </article>
          </section>
        ) : null}

        <button
          type="button"
          className={`primary-assistant-button ${primaryKind}-command-button ${voiceDetected ? "voice-active" : ""}`}
          onClick={handlePrimaryAction}
          disabled={ocrBusy || cameraMode === "opening"}
          aria-label={primaryLabel}
          style={primaryButtonStyle}
        >
          <span className="primary-button-icon" aria-hidden="true">
            {primaryKind === "camera" ? <span className="camera-icon-shape" /> : <span className="mic-icon-shape" />}
          </span>
          {primaryKind === "mic" ? (
            <span className="mic-level-bars" aria-hidden="true">
              <span />
              <span />
              <span />
            </span>
          ) : null}
          <span className="visually-hidden">{primaryLabel}</span>
        </button>

        {!adminMode && liveReplyText ? (
          <section className="live-dialog below-mic" aria-label="오디스 답변" aria-live="polite">
            <article className={`live-card ${liveReplySender} ${liveReplySize}`}>
              <p>{liveReplyText}</p>
            </article>
          </section>
        ) : null}

        {cameraMode === "error" ? (
          <button type="button" className="quiet-button" onClick={() => fileInputRef.current?.click()}>
            사진 선택하기
          </button>
        ) : null}
        {showCamera ? (
          <button type="button" className="quiet-button" onClick={() => closeCameraSession("카메라를 닫았습니다.")}>
            카메라 닫기
          </button>
        ) : null}

        <input
          ref={fileInputRef}
          className="visually-hidden"
          type="file"
          accept="image/*"
          capture="environment"
          onChange={handleOcrFileChange}
        />

        {adminMode && hasLiveDialog ? (
          <section className="recent-dialog" aria-label="최근 대화" aria-live="polite">
            {liveUserText ? (
              <article className="recent-card user">
                <span>나</span>
                <p>{liveUserText}</p>
              </article>
            ) : null}
            {liveReplyText ? (
              <article className={`recent-card ${liveReplySender}`}>
                <span>{senderLabel(liveReplySender)}</span>
                <p>{liveReplyText}</p>
              </article>
            ) : null}
          </section>
        ) : null}

        <div className="assistant-footer-actions">
          <button
            type="button"
            className={`icon-utility-button keyboard-toggle ${manualOpen ? "active" : ""}`}
            onClick={() => setManualOpen((value) => !value)}
            aria-label="직접 입력"
            title="직접 입력"
          >
            <span className="keyboard-icon-shape" aria-hidden="true" />
          </button>
          {speaking ? (
            <button type="button" className="icon-utility-button stop-voice" onClick={stopTts} aria-label="음성 중지" title="음성 중지">
              <span className="stop-icon-shape" aria-hidden="true" />
            </button>
          ) : null}
        </div>

        {manualOpen ? (
          <form className="chat-input compact" onSubmit={handleManualSubmit}>
            <input
              value={manualText}
              onChange={(event) => setManualText(event.target.value)}
              placeholder="말씀을 적어 주세요"
              aria-label="대화 입력"
            />
            <button type="submit" disabled={!manualText.trim()}>전송</button>
          </form>
        ) : null}
      </main>

      {adminMode ? (
        <aside className="admin-drawer">
          <section className="panel debug-panel">
            <h3>관리자 디버그</h3>
            <div className="admin-toggles">
              <label className="toggle-row">
                <input
                  type="checkbox"
                  checked={voiceArmed}
                  onChange={(event) => {
                    if (event.target.checked) {
                      startAssistant();
                    } else {
                      stopListening();
                    }
                  }}
                />
                자동 대기
              </label>
              <label className="toggle-row">
                <input type="checkbox" checked={ttsEnabled} onChange={(event) => setTtsEnabled(event.target.checked)} />
                TTS
              </label>
              <label className="toggle-row">
                <input
                  type="checkbox"
                  checked={fillerTtsEnabled}
                  onChange={(event) => setFillerTtsEnabled(event.target.checked)}
                />
                대기 안내
              </label>
            </div>
            <div className="toolbar-actions">
              <button type="button" className="ghost-button" onClick={connectWebSocket}>재연결</button>
              <button type="button" className="ghost-button" onClick={startNewUser}>새 사용자</button>
              <button type="button" className="ghost-button" onClick={exportSessionLog}>로그 내보내기</button>
            </div>
            <dl>
              <dt>speaker_id</dt>
              <dd>{speakerId}</dd>
              <dt>session_id</dt>
              <dd>{sessionId}</dd>
              <dt>token</dt>
              <dd>{token ? "configured" : "empty/local"}</dd>
            </dl>
          </section>

          <section className="panel timeline-panel">
            <h3>대화 로그</h3>
            <div className="timeline" aria-label="대화 타임라인">
              {chronologicalMessages.length === 0 ? (
                <div className="empty-state">
                  <strong>아직 대화가 없습니다.</strong>
                </div>
              ) : (
                chronologicalMessages.map((message) => (
                  <MessageItem
                    key={message.id}
                    message={message}
                    adminMode={adminMode}
                  />
                ))
              )}
            </div>
          </section>
        </aside>
      ) : null}
    </section>
  );
}

function MessageItem({
  message,
  adminMode,
}: {
  message: AssistantMessage;
  adminMode: boolean;
}) {
  const rawPayload = (message.raw && typeof message.raw === "object" ? message.raw : {}) as WsPayload;
  return (
    <article className={`message ${message.sender}`}>
      <header>
        <strong>{senderLabel(message.sender)}</strong>
        <span>{formatTime(message.createdAt)}</span>
      </header>
      <p>{message.text || "(내용 없음)"}</p>
      {adminMode ? (
        <div className="message-meta">
          {message.responseType ? <span>{message.responseType}</span> : null}
          {message.turnId ? <span>turn: {message.turnId}</span> : null}
          {message.fastPath ? <span>fast: {message.fastPath}</span> : null}
          {message.stage ? <span>stage: {message.stage}</span> : null}
          {message.latency?.firstMessageMs ? <span>first {message.latency.firstMessageMs}ms</span> : null}
          {message.latency?.finalResponseMs ? <span>final {message.latency.finalResponseMs}ms</span> : null}
          {typeof rawPayload.server_elapsed_ms === "number" ? (
            <span>server {rawPayload.server_elapsed_ms}ms</span>
          ) : null}
          {typeof rawPayload.ws_elapsed_ms === "number" ? <span>ws {rawPayload.ws_elapsed_ms}ms</span> : null}
        </div>
      ) : null}
      {adminMode && message.raw ? (
        <details className="raw-json">
          <summary>raw JSON</summary>
          <pre>{JSON.stringify(message.raw, null, 2)}</pre>
        </details>
      ) : null}
    </article>
  );
}

function payloadText(payload: WsPayload): string {
  return String(payload.response_text || payload.text || payload.message || payload.reason || "");
}

function captionSizeClass(text: string): string {
  const length = Array.from(text.trim()).length;
  if (length <= 14) return "caption-short";
  if (length <= 34) return "caption-medium";
  if (length <= 78) return "caption-long";
  return "caption-xlong";
}

function payloadSender(payload: WsPayload): AssistantMessage["sender"] {
  if (payload.type === "filler") {
    return "filler";
  }
  if (payload.type === "error") {
    return "warning";
  }
  if (payload.type === "ignored" || payload.type === "session_closed") {
    return "system";
  }
  return "odiss";
}

function payloadRequestsCameraClose(payload: WsPayload, text: string): boolean {
  if (payload.ui_action === "close_camera") {
    return true;
  }
  const responseType = String(payload.response_type || payload.type || "");
  if (responseType === "ocr_cancelled" || payload.fast_path === "assistant_camera_cancel") {
    return true;
  }
  return /사진\s*확인을?\s*중단|카메라를?\s*닫|촬영을?\s*중단/.test(text);
}

function isLiveReplyMessage(message: AssistantMessage, adminMode: boolean): boolean {
  if (message.sender === "odiss" || message.sender === "warning") {
    return !isIdleTimeoutMessage(message);
  }
  if (adminMode && message.sender === "system") {
    return !isIdleTimeoutMessage(message);
  }
  return false;
}

function isIdleTimeoutMessage(message: AssistantMessage): boolean {
  const raw = (message.raw && typeof message.raw === "object" ? message.raw : {}) as WsPayload;
  return message.responseType === "session_closed" ||
    message.reason === "idle_timeout" ||
    raw.reason === "idle_timeout" ||
    message.text === "idle_timeout";
}

function isFinalPayload(payload: WsPayload): boolean {
  return !["filler", "pong"].includes(String(payload.type || ""));
}

function isPhotoIntent(text: string): boolean {
  const compact = text.replace(/\s+/g, "");
  if (isCameraDismissIntent(text)) {
    return false;
  }
  const explicitCamera = /(카메라|사진|촬영|OCR|오씨알).*(켜|열|찍|촬영|확인|읽|인식|보여|올릴|업로드|스캔)/i;
  const explicitCameraReverse = /(켜|열|찍|촬영|확인|읽|인식|보여|올릴|업로드|스캔).*(카메라|사진|촬영|OCR|오씨알)/i;
  const documentCapture = /(약봉투|처방전|약통|약사진).*(찍|촬영|사진|보여|읽|확인|인식|스캔|올릴|업로드)/i;
  const documentCaptureReverse = /(찍|촬영|사진|보여|읽|확인|인식|스캔|올릴|업로드).*(약봉투|처방전|약통|약사진)/i;
  return (
    explicitCamera.test(compact) ||
    explicitCameraReverse.test(compact) ||
    documentCapture.test(compact) ||
    documentCaptureReverse.test(compact)
  );
}

function isCameraDismissIntent(text: string): boolean {
  const compact = text.replace(/\s+/g, "");
  return /(카메라|사진|촬영|OCR|오씨알).*(꺼|끄|닫|치워|취소|그만|안해|안할|안찍|찍지마|찍지말|필요없|됐)/i.test(compact) ||
    /(꺼|끄|닫|치워|취소|그만|안해|안할|안찍|찍지마|찍지말|필요없|됐).*(카메라|사진|촬영|OCR|오씨알)/i.test(compact) ||
    /사진안찍/i.test(compact) ||
    /^(취소|그만|됐어|아니야|아니|안해|필요없어|닫아|닫어|꺼|꺼줘|치워)$/.test(compact);
}

function mentionsCameraSurface(text: string): boolean {
  return /(카메라|사진|촬영|OCR|오씨알|약봉투|처방전|약통)/i.test(text.replace(/\s+/g, ""));
}

function showCameraPanel(cameraMode: CameraMode, ocrPreview: string, ocrBusy: boolean): boolean {
  return cameraMode !== "idle" || Boolean(ocrPreview) || ocrBusy;
}

function isLikelyAssistantEcho(heard: string, spoken: string): boolean {
  const heardNorm = normalizeSpeechText(heard);
  const spokenNorm = normalizeSpeechText(spoken);
  if (!heardNorm || !spokenNorm || heardNorm.length < 4) {
    return false;
  }
  return spokenNorm.includes(heardNorm) || heardNorm.includes(spokenNorm.slice(0, heardNorm.length));
}

function normalizeSpeechText(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, "")
    .trim();
}

function downloadJson(filename: string, value: unknown) {
  const blob = new Blob([JSON.stringify(value, null, 2)], {
    type: "application/json;charset=utf-8",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function compactTimestamp(iso: string): string {
  return iso.replace(/[:.]/g, "-");
}

function currentAssistantMode(input: {
  cameraMode: CameraMode;
  latestMessage?: AssistantMessage;
  listening: boolean;
  ocrBusy: boolean;
  speaking: boolean;
  status: ConnectionStatus;
  voiceArmed: boolean;
}): AssistantMode {
  if (input.status === "error") return "error";
  if (input.ocrBusy) return "ocr_processing";
  if (input.cameraMode === "opening" || input.cameraMode === "ready" || input.cameraMode === "error") {
    return "camera_ready";
  }
  if (input.speaking) return "speaking";
  if (input.listening) return "listening";
  if (input.latestMessage?.sender === "filler") return "thinking";
  if (input.voiceArmed) return "listening";
  return "idle";
}

function assistantModeCopy(
  mode: AssistantMode,
  input: {
    cameraMessage: string;
    interimText: string;
    speechSupported: boolean;
    status: ConnectionStatus;
    voiceArmed: boolean;
  },
): { kicker: string; title: string; subtitle: string } {
  if (mode === "listening") {
    return {
      kicker: input.voiceArmed ? "마이크 대기 중" : "듣고 있어요",
      title: input.interimText || "편하게 말씀하세요",
      subtitle: "말이 끝나면 제가 알아서 잘라서 확인합니다.",
    };
  }
  if (mode === "thinking") {
    return {
      kicker: "확인 중",
      title: "잠시만요",
      subtitle: "필요한 기록과 약 정보를 보고 있습니다.",
    };
  }
  if (mode === "camera_ready") {
    return {
      kicker: "사진 확인",
      title: "약봉투를 화면에 맞춰주세요",
      subtitle: input.cameraMessage,
    };
  }
  if (mode === "ocr_processing") {
    return {
      kicker: "사진 확인 중",
      title: "사진을 읽고 있어요",
      subtitle: "약 이름이 보이는지 확인한 뒤 대화로 알려드리겠습니다.",
    };
  }
  if (mode === "speaking") {
    return {
      kicker: "답변 중",
      title: "말씀드리고 있어요",
      subtitle: "중간에 말씀하시면 제가 멈추고 바로 들을게요.",
    };
  }
  if (mode === "error") {
    return {
      kicker: "연결 확인",
      title: "잠시 문제가 있어요",
      subtitle: input.status === "error" ? "연결을 다시 시도하고 있습니다." : "다시 한 번 시도해 주세요.",
    };
  }
  return {
    kicker: input.speechSupported ? "오디스 대기" : "직접 입력 가능",
    title: "오디스에게 말씀하세요",
    subtitle: input.speechSupported
      ? "가운데 마이크를 한 번 누르면, 이후에는 말이 끝날 때마다 오디스가 알아서 듣습니다."
      : "이 브라우저는 음성 입력을 지원하지 않아 직접 입력으로 대화합니다.",
  };
}

function primaryActionLabel(input: {
  cameraMode: CameraMode;
  listening: boolean;
  ocrBusy: boolean;
  speechSupported: boolean;
  voiceArmed: boolean;
}): string {
  if (input.ocrBusy) return "확인 중";
  if (input.cameraMode === "opening") return "카메라 준비 중";
  if (input.cameraMode === "ready") return "촬영하기";
  if (input.cameraMode === "error") return "사진 선택하기";
  if (input.voiceArmed || input.listening) return "듣기 중지";
  return input.speechSupported ? "오디스 시작" : "직접 입력하기";
}

function primaryActionKind(input: { cameraMode: CameraMode; ocrBusy: boolean }): "mic" | "camera" {
  if (input.ocrBusy || input.cameraMode === "opening" || input.cameraMode === "ready" || input.cameraMode === "error") {
    return "camera";
  }
  return "mic";
}

function senderLabel(sender: AssistantMessage["sender"]): string {
  if (sender === "user") return "YOU";
  if (sender === "filler") return "대기 안내";
  if (sender === "warning") return "알림";
  if (sender === "system") return "시스템";
  return "ODISS";
}

function statusLabel(status: ConnectionStatus): string {
  if (status === "connected") return "연결됨";
  if (status === "connecting") return "연결 중";
  if (status === "error") return "오류";
  return "닫힘";
}

function formatTime(iso: string): string {
  return new Intl.DateTimeFormat("ko-KR", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date(iso));
}
