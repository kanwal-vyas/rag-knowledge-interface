/**
 * Voice streaming client.
 *
 * Captures microphone audio via AudioWorklet, resamples to 16kHz mono
 * linear16 PCM, and streams to the backend WebSocket proxy.
 * Receives partial/final transcripts from Sarvam in real time.
 *
 * Architecture:
 *   Microphone → AudioWorklet → PCM chunks → WebSocket → Backend → Sarvam
 *   Sarvam → Backend → WebSocket → partial/final transcripts → UI
 */

export type StreamEvent =
  | { event: "connected"; latency_ws_connect_ms: number }
  | { event: "vad.speech_start" }
  | { event: "vad.speech_end" }
  | { event: "transcript.partial"; text: string; is_stream_final: boolean }
  | { event: "transcript.final"; text: string; language?: string; language_confidence?: number }
  | { event: "session.end"; audio_duration_s?: number }
  | { event: "error"; message: string; fatal?: boolean }
  | { event: "latency"; total_session_ms: number; audio_bytes_sent: number };

export interface VoiceStreamCallbacks {
  onPartial?: (text: string) => void;
  onFinal?: (text: string, language?: string) => void;
  onSpeechStart?: () => void;
  onSpeechEnd?: () => void;
  onConnected?: (latencyMs: number) => void;
  onError?: (message: string, fatal: boolean) => void;
  onLatency?: (totalMs: number, audioBytes: number) => void;
  onStateChange?: (state: "idle" | "connecting" | "streaming" | "processing" | "done") => void;
}

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");
const WS_BASE = API_BASE.replace(/^http/, "ws");

export class VoiceStreamClient {
  private ws: WebSocket | null = null;
  private audioCtx: AudioContext | null = null;
  private workletNode: AudioWorkletNode | null = null;
  private sourceNode: MediaStreamAudioSourceNode | null = null;
  private stream: MediaStream | null = null;
  private callbacks: VoiceStreamCallbacks;
  private _state: "idle" | "connecting" | "streaming" | "processing" | "done" = "idle";
  private _lang: string = "en";

  constructor(callbacks: VoiceStreamCallbacks) {
    this.callbacks = callbacks;
  }

  get state() { return this._state; }

  private setState(s: typeof this._state) {
    this._state = s;
    this.callbacks.onStateChange?.(s);
  }

  async start(lang: string = "en") {
    if (this._state !== "idle") return;
    this._lang = lang;
    this.setState("connecting");

    try {
      // 1. Get microphone stream.
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // 2. Create AudioContext at 16kHz for minimal resampling overhead.
      this.audioCtx = new AudioContext({ sampleRate: 16000 });

      // 3. Load AudioWorklet processor.
      await this.audioCtx.audioWorklet.addModule("/pcm-processor.js");

      // 4. Create nodes.
      this.sourceNode = this.audioCtx.createMediaStreamSource(this.stream);
      this.workletNode = new AudioWorkletNode(this.audioCtx, "pcm-processor");

      // 5. Listen for PCM chunks from the worklet.
      this.workletNode.port.onmessage = (e: MessageEvent<ArrayBuffer>) => {
        if (this.ws?.readyState === WebSocket.OPEN) {
          this.ws.send(e.data);
        }
      };

      // 6. Connect: mic → worklet.
      this.sourceNode.connect(this.workletNode);
      // Don't connect to destination (no feedback loop).

      // 7. Open WebSocket to backend.
      this.ws = new WebSocket(`${WS_BASE}/voice/stream`);

      this.ws.onopen = () => {
        // Send start event with language.
        this.ws!.send(JSON.stringify({ event: "start", lang: this._lang }));
      };

      this.ws.onmessage = (e: MessageEvent) => {
        try {
          const data: StreamEvent = JSON.parse(e.data);
          this.handleEvent(data);
        } catch { /* ignore parse errors */ }
      };

      this.ws.onclose = () => {
        this.stop();
      };

      this.ws.onerror = () => {
        this.callbacks.onError?.("WebSocket connection failed", true);
        this.stop();
      };

    } catch (err: any) {
      this.callbacks.onError?.(err.message || "Failed to start streaming", true);
      this.stop();
    }
  }

  private handleEvent(data: StreamEvent) {
    switch (data.event) {
      case "connected":
        this.setState("streaming");
        this.callbacks.onConnected?.(data.latency_ws_connect_ms);
        break;
      case "vad.speech_start":
        this.callbacks.onSpeechStart?.();
        break;
      case "vad.speech_end":
        this.callbacks.onSpeechEnd?.();
        this.setState("processing");
        break;
      case "transcript.partial":
        this.callbacks.onPartial?.(data.text);
        break;
      case "transcript.final":
        this.callbacks.onFinal?.(data.text, data.language);
        this.setState("done");
        break;
      case "session.end":
        this.stop();
        break;
      case "error":
        this.callbacks.onError?.(data.message, data.fatal ?? false);
        if (data.fatal) this.stop();
        break;
      case "latency":
        this.callbacks.onLatency?.(data.total_session_ms, data.audio_bytes_sent);
        break;
    }
  }

  /** Signal end of speech — Sarvam will finalize the transcript. */
  stopRecording() {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ event: "stop" }));
    }
    // Stop the worklet (stops sending audio but keeps WS open for transcript).
    this.workletNode?.port.postMessage("stop");
  }

  /** Full teardown: stop mic, close WS, release resources. */
  stop() {
    try { this.workletNode?.port.postMessage("stop"); } catch { /* */ }
    try { this.sourceNode?.disconnect(); } catch { /* */ }
    try { this.workletNode?.disconnect(); } catch { /* */ }
    try { this.audioCtx?.close(); } catch { /* */ }
    try { this.stream?.getTracks().forEach(t => t.stop()); } catch { /* */ }
    try { this.ws?.close(); } catch { /* */ }

    this.ws = null;
    this.audioCtx = null;
    this.workletNode = null;
    this.sourceNode = null;
    this.stream = null;

    if (this._state !== "idle") {
      this.setState("idle");
    }
  }
}
