// Signal Room style reminder: answer-first editorial hierarchy, mineral dark surfaces, saffron signal accents, and honest instrument-like state changes.

import { useEffect, useRef, useState } from "react";
import {
  Activity,
  ArrowUpRight,
  Check,
  ChevronDown,
  CircleHelp,
  Clock3,
  Languages,
  Mic,
  MicOff,
  Radio,
  RotateCcw,
  Send,
  ShieldCheck,
  Sparkles,
  Square,
  Volume2,
  X,
} from "lucide-react";
import { getHealth, queryText, queryVoice, type Latency, type QueryResponse, type VoiceQueryResponse } from "@/lib/api";

const examples = [
  "What is the purpose of the European Union?",
  "How does photosynthesis convert light into energy?",
  "भारत में मानसून कैसे बनता है।",
];

type UiState = "idle" | "recording" | "processing" | "complete" | "error";

type Result = (QueryResponse & { transcript?: string; detected_language?: string }) | null;

const formatMs = (value: number | null | undefined) => value == null ? null : `${Math.round(value)} ms`;

/** Detect whether the backend reported a generation/provider error. */
function hasGenerationError(latency: Latency): boolean {
  return typeof latency.error === "string" && latency.error.length > 0;
}

function LatencyDetails({ latency }: { latency: Latency }) {
  const rows = [
    ["Retrieval", latency.sparse_retrieval_ms],
    ["Context assembly", latency.context_assembly_ms],
    ["Generation", latency.generation_ms],
    ["STT", latency.stt_ms],
    ["Total", latency.total_ms],
  ].filter(([, value]) => value != null) as [string, number][];
  const hasError = typeof latency.error === "string" && latency.error.length > 0;
  if (!rows.length && !hasError) return null;
  return (
    <details className="latency-details">
      <summary><Clock3 size={14} /> Pipeline performance <ChevronDown size={14} /></summary>
      <div className="latency-grid">
        {rows.map(([label, value]) => <span key={label}><small>{label}</small><strong>{formatMs(value)}</strong></span>)}
        {hasError && <span className="latency-error"><small>Error</small><strong>Provider unavailable</strong></span>}
      </div>
    </details>
  );
}

function SignalLogo() {
  return <div className="logo-mark" aria-label="HH RAG"><span /><i /><b /></div>;
}

const voiceLanguages = [
  { value: "", label: "Auto-detect" },
  { value: "en", label: "English" },
  { value: "hi", label: "Hindi" },
  { value: "gu", label: "Gujarati" },
  { value: "bn", label: "Bengali" },
  { value: "ta", label: "Tamil" },
  { value: "te", label: "Telugu" },
  { value: "mr", label: "Marathi" },
  { value: "kn", label: "Kannada" },
  { value: "ml", label: "Malayalam" },
  { value: "pa", label: "Punjabi" },
];

export default function Home() {
  const [query, setQuery] = useState("");
  const [state, setState] = useState<UiState>("idle");
  const [result, setResult] = useState<Result>(null);
  const [error, setError] = useState("");
  const [recordSeconds, setRecordSeconds] = useState(0);
  const [health, setHealth] = useState("checking");
  const [voiceLang, setVoiceLang] = useState("en");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | undefined>(undefined);
  const recordStartRef = useRef<number>(0);
  const sessionRef = useRef<number>(0);
  const lastBlobRef = useRef<Blob | null>(null);
  const audioUrlRef = useRef<string | null>(null);
  const [diagResult, setDiagResult] = useState("");
  const [diagBusy, setDiagBusy] = useState(false);
  const [canPlay, setCanPlay] = useState(false);
  const [micTestInfo, setMicTestInfo] = useState<string[]>([]);
  const [micTestStream, setMicTestStream] = useState<MediaStream | null>(null);
  const [micRms, setMicRms] = useState(0);
  const micTestMonitorRef = useRef<HTMLAudioElement | null>(null);
  const micAudioCtxRef = useRef<AudioContext | null>(null);
  const micRmsIntervalRef = useRef<number | undefined>(undefined);

  useEffect(() => {
    getHealth().then((data) => setHealth(data.status === "ok" ? "online" : "degraded")).catch(() => setHealth("offline"));
    return () => {
      sessionRef.current++; // Invalidate any in-flight recording session.
      if (timerRef.current) window.clearInterval(timerRef.current);
      const rec = mediaRecorderRef.current;
      if (rec && rec.state !== "inactive") {
        try { rec.stop(); } catch { /* already stopped */ }
      }
      // Release audio URL.
      if (audioUrlRef.current) { URL.revokeObjectURL(audioUrlRef.current); audioUrlRef.current = null; }
    };
  }, []);

  const submitText = async () => {
    const trimmed = query.trim();
    if (!trimmed || state === "processing" || state === "recording") {
      if (!trimmed) { setError("Add a question before sending it to the knowledge base."); setState("error"); }
      return;
    }
    setError(""); setResult(null); setState("processing");
    try {
      setResult(await queryText(trimmed)); setState("complete");
    } catch (err) {
      setError(err instanceof Error ? err.message : "The knowledge service could not be reached."); setState("error");
    }
  };

  const startRecording = async () => {
    setError("");
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      setError("Voice capture is not supported in this browser. Try a current Chrome, Edge, or Safari browser."); setState("error"); return;
    }
    try {
      // Stop any previous recorder cleanly before starting a new one.
      const prevRec = mediaRecorderRef.current;
      if (prevRec && prevRec.state !== "inactive") {
        try { prevRec.requestData(); prevRec.stop(); } catch { /* already stopped */ }
      }

      // Simplest possible mic acquisition - no constraints, no processing.
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // Log MediaStream diagnostics.
      const tracks = stream.getTracks();
      const audioTracks = stream.getAudioTracks();
      console.log(`[Voice] MICROPHONE`);
      console.log(`[Voice] tracks: ${tracks.length}, audioTracks: ${audioTracks.length}`);
      tracks.forEach((t, i) => {
        const s = t.getSettings();
        console.log(`[Voice]   track[${i}]: kind=${t.kind} label=${t.label} readyState=${t.readyState} enabled=${t.enabled} muted=${t.muted}`);
        console.log(`[Voice]   settings: sampleRate=${s.sampleRate} channelCount=${s.channelCount} deviceId=${s.deviceId}`);
      });
      console.log(`[Voice] stream.active: ${stream.active}`);

      const recorder = new MediaRecorder(stream);
      console.log(`[Voice] recorder.mimeType: ${recorder.mimeType}`);

      // Clear old chunks and increment session BEFORE registering handlers.
      chunksRef.current = [];
      const sessionId = ++sessionRef.current;

      console.log(`[Voice] --- SESSION ${sessionId} START ---`);

      let chunkIndex = 0;
      recorder.ondataavailable = (event) => {
        // Guard: only accept chunks from the current session.
        if (sessionId !== sessionRef.current) return;
        if (event.data.size) {
          chunksRef.current.push(event.data);
          chunkIndex++;
          console.log(`[Voice] [session=${sessionId}] chunk #${chunkIndex}: ${event.data.size} bytes (total: ${chunksRef.current.length})`);
        }
      };
      recorder.onstop = () => {
        // CRITICAL: Capture chunks and session immediately - do NOT read
        // chunksRef.current later, as a new session may have cleared it.
        const capturedChunks = [...chunksRef.current];
        const capturedSession = sessionId;
        stream.getTracks().forEach((track) => track.stop());
        if (timerRef.current) window.clearInterval(timerRef.current);
        const wallMs = Date.now() - recordStartRef.current;

        console.log(`[Voice] [session=${capturedSession}] RECORDING SUMMARY`);
        console.log(`[Voice] [session=${capturedSession}] recorder.mimeType: ${recorder.mimeType}`);
        console.log(`[Voice] [session=${capturedSession}] captured chunks: ${capturedChunks.length}`);
        console.log(`[Voice] [session=${capturedSession}] chunk sizes: [${capturedChunks.map(c => c.size).join(", ")}]`);
        console.log(`[Voice] [session=${capturedSession}] wall-clock duration: ${wallMs}ms`);

        // If a newer recording has already started, discard this one.
        if (capturedSession !== sessionRef.current) {
          console.log(`[Voice] [session=${capturedSession}] STALE - discarding (current session: ${sessionRef.current})`);
          return;
        }

        const blob = new Blob(capturedChunks, { type: recorder.mimeType || "audio/webm" });
        console.log(`[Voice] [session=${capturedSession}] blob.size: ${blob.size} bytes`);
        console.log(`[Voice] [session=${capturedSession}] blob.type: ${blob.type}`);

        if (!blob.size) {
          console.log(`[Voice] [session=${capturedSession}] EMPTY BLOB - no audio captured`);
          setError("No audio was captured. Please try again."); setState("error");
          return;
        }

        // Store blob ref for playback and /diagnose/audio.
        lastBlobRef.current = blob;
        if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current);
        audioUrlRef.current = URL.createObjectURL(blob);
        setCanPlay(true);

        setState("processing");
        // Fire-and-forget async processing - session guard prevents stale results.
        processVoiceBlob(blob, capturedSession, capturedChunks.length, wallMs);
      };

      recordStartRef.current = Date.now();
      recorder.start();
      mediaRecorderRef.current = recorder;
      setRecordSeconds(0);
      setState("recording");
      timerRef.current = window.setInterval(() => setRecordSeconds((value) => value + 1), 1000);
    } catch {
      setError("Microphone access was denied. Allow microphone access in your browser and try again."); setState("error");
    }
  };

  const processVoiceBlob = async (blob: Blob, sessionId: number, chunkCount: number, wallMs: number) => {
    try {
      console.log(`[Voice] [session=${sessionId}] uploading ${blob.size} bytes (${chunkCount} chunks, ${wallMs}ms)`);
      const voiceResult = await queryVoice(blob, voiceLang || undefined, sessionId);
      // Guard: only apply results if this session is still current.
      if (sessionId !== sessionRef.current) {
        console.log(`[Voice] [session=${sessionId}] result discarded - session superseded`);
        return;
      }
      console.log(`[Voice] [session=${sessionId}] result: transcript="${voiceResult.transcript}"`);
      setResult(voiceResult);
      setQuery(voiceResult.transcript || "");
      setState("complete");
    } catch (err) {
      if (sessionId !== sessionRef.current) return;
      console.log(`[Voice] [session=${sessionId}] error: ${err}`);
      setError(err instanceof Error ? err.message : "Voice processing failed. Please try text instead.");
      setState("error");
    }
  };

  const runDiag = async () => {
    const blob = lastBlobRef.current;
    if (!blob) { setDiagResult("No recording available. Record something first."); return; }
    setDiagBusy(true); setDiagResult("Sending to /diagnose/audio ...\n");
    try {
      const form = new FormData();
      form.append("file", blob, "diag.webm");
      form.append("lang", voiceLang || "en");
      const resp = await fetch(`${(import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "")}/diagnose/audio`, { method: "POST", body: form });
      const json = await resp.json();
      setDiagResult(JSON.stringify(json, null, 2));
    } catch (e) {
      setDiagResult(`Error: ${e}`);
    } finally {
      setDiagBusy(false);
    }
  };

  // ── MIC TEST: Web Audio RMS + live playback ──
  const stopMicTest = () => {
    if (micRmsIntervalRef.current) { clearInterval(micRmsIntervalRef.current); micRmsIntervalRef.current = undefined; }
    if (micAudioCtxRef.current) { micAudioCtxRef.current.close().catch(() => {}); micAudioCtxRef.current = null; }
    if (micTestStream) {
      micTestStream.getTracks().forEach(t => t.stop());
      setMicTestStream(null);
    }
    if (micTestMonitorRef.current) {
      micTestMonitorRef.current.srcObject = null;
    }
    setMicTestInfo([]);
    setMicRms(0);
  };

  const testMic = async () => {
    stopMicTest();
    const lines: string[] = [];
    const push = (s: string) => { lines.push(s); console.log(`[MicTest] ${s}`); };

    push("=== TEST 1: MediaStream ===");
    push(`navigator.mediaDevices: ${!!navigator.mediaDevices}`);
    push(`getUserMedia: ${!!navigator.mediaDevices?.getUserMedia}`);

    try {
      // Simplest possible request - no constraints.
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const tracks = stream.getAudioTracks();
      push(`getUserMedia: OK`);
      push(`audio tracks: ${tracks.length}`);
      tracks.forEach((t, i) => {
        const s = t.getSettings();
        push(`track[${i}]: kind=${t.kind}`);
        push(`  label: "${t.label}"`);
        push(`  readyState: ${t.readyState}`);
        push(`  enabled: ${t.enabled}`);
        push(`  muted: ${t.muted}`);
        push(`  settings: ${JSON.stringify(s)}`);
      });
      push(`stream.active: ${stream.active}`);

      // Enumerate devices.
      const devices = await navigator.mediaDevices.enumerateDevices();
      const audioInputs = devices.filter(d => d.kind === "audioinput");
      push(`\navailable audioinput devices: ${audioInputs.length}`);
      audioInputs.forEach((d, i) => {
        push(`  [${i}] "${d.label}" id=${d.deviceId.slice(0, 20)}...`);
      });

      setMicTestStream(stream);
      setMicTestInfo([...lines]);

      // TEST 2: Web Audio API RMS signal analysis.
      push("\n=== TEST 2: Web Audio RMS ===");
      const ctx = new AudioContext();
      micAudioCtxRef.current = ctx;
      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      const floatData = new Float32Array(analyser.fftSize);
      push(`AudioContext sampleRate: ${ctx.sampleRate}`);
      push(`Analyser fftSize: ${analyser.fftSize}`);
      push("Speak now - RMS updates every 200ms below...");
      micRmsIntervalRef.current = window.setInterval(() => {
        analyser.getFloatTimeDomainData(floatData);
        let sum = 0;
        for (let i = 0; i < floatData.length; i++) sum += floatData[i] * floatData[i];
        const rms = Math.sqrt(sum / floatData.length);
        setMicRms(rms);
      }, 200);
      setMicTestInfo([...lines]);

      // TEST 3: Audio element playback.
      push("\n=== TEST 3: Audio playback ===");
      const audio = new Audio();
      audio.srcObject = stream;
      audio.autoplay = true;
      audio.muted = false;
      audio.volume = 1.0;
      await audio.play().catch((e: any) => {
        push(`Audio.play() FAILED: ${e.name}: ${e.message}`);
        push(`Try: chrome://settings/content/sound - ensure not blocked`);
      });
      push(`Audio element: ${audio.paused ? "PAUSED" : "PLAYING"}`);
      push(`You should hear yourself NOW through speakers/headphones`);
      micTestMonitorRef.current = audio;
      setMicTestInfo([...lines]);

    } catch (e: any) {
      push(`\ngetUserMedia FAILED`);
      push(`error.name: ${e.name}`);
      push(`error.message: ${e.message}`);
      if (e.name === "NotAllowedError") push("-> Microphone permission DENIED. Check: chrome://settings/content/microphone");
      else if (e.name === "NotFoundError") push("-> No microphone found. Check: chrome://settings/content/microphone");
      else if (e.name === "NotReadableError") push("-> Microphone in use by another app (Teams, Zoom, etc.)");
      else if (e.name === "OverconstrainedError") push("-> Constraints cannot be satisfied");
      else if (e.name === "SecurityError") push("-> Must be served over HTTPS or localhost");
      setMicTestInfo([...lines]);
    }
  };

  const stopRecording = () => {
    const rec = mediaRecorderRef.current;
    if (rec && rec.state !== "inactive") {
      try {
        rec.requestData();
        rec.stop();
      } catch {
        // Recorder may already be stopped.
      }
    }
  };
  const reset = () => { setQuery(""); setResult(null); setError(""); setState("idle"); setRecordSeconds(0); setTimeout(() => textareaRef.current?.focus(), 0); };
  const isBusy = state === "processing";

  return (
    <main className="app-shell">
      <aside className="signal-rail">
        <div className="rail-top"><SignalLogo /><span className="rail-version">HH / 01</span></div>
        <div className="rail-middle">
          <div className="rail-label"><span className="vertical-label">VOICE-ENABLED RAG</span></div>
          <div className="rail-meter"><span className="meter-fill" /></div>
          <span className="rail-caption">Ask clearly.<br />Ground deeply.</span>
        </div>
        <div className="rail-bottom">
          <div className={`health-dot ${health}`} /><span>{health === "checking" ? "Checking service" : health === "online" ? "Service online" : "Service unavailable"}</span>
          <span className="rail-index">2026</span>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div className="brand-lockup"><span className="brand-name"><b>HH</b><i>RAG</i></span><span className="brand-rule" /><span className="brand-detail">KNOWLEDGE INTERFACE</span></div>
          <div className="top-actions"><span className="secure-readout"><ShieldCheck size={14} /> Backend-only keys</span></div>
        </header>

        <div className="workspace-grid">
          <section className="prompt-column">
            <div className="eyebrow"><span className="eyebrow-line" /> INPUT / 01</div>
            <h1>Ask the<br /><em>knowledge base.</em></h1>
            <p className="intro">One question, two ways in. Speak or type and receive an answer grounded in retrieved knowledge.</p>

            <div className={`composer ${state === "recording" ? "is-recording" : ""} ${isBusy ? "is-busy" : ""}`}>
              <div className="composer-head"><span>{state === "recording" ? "Listening" : state === "processing" ? "Working through the pipeline" : "Your question"}</span><span className="composer-mode">{state === "recording" ? `${String(Math.floor(recordSeconds / 60)).padStart(2, "0")}:${String(recordSeconds % 60).padStart(2, "0")}` : "TEXT / VOICE"}</span></div>
              <textarea ref={textareaRef} value={query} onChange={(event) => { setQuery(event.target.value); if (state === "error") setState("idle"); }} onKeyDown={(event) => { if ((event.metaKey || event.ctrlKey) && event.key === "Enter") submitText(); }} placeholder="What would you like to understand?" disabled={state === "recording" || isBusy} aria-label="Ask the knowledge base" />
              <div className="composer-actions">
                <div className="voice-controls">
                  {state === "recording" ? <button className="mic-button recording" onClick={stopRecording} aria-label="Stop recording"><Square size={18} fill="currentColor" /><span className="pulse-ring" /></button> : <button className="mic-button" onClick={startRecording} disabled={isBusy} aria-label="Record a voice question"><Mic size={20} /></button>}
                  <select className="voice-lang-select" value={voiceLang} onChange={(e) => setVoiceLang(e.target.value)} disabled={isBusy} aria-label="Voice language">
                    {voiceLanguages.map((l) => <option key={l.value} value={l.value}>{l.label}</option>)}
                  </select>
                </div>
                <button className="send-button" onClick={submitText} disabled={!query.trim() || isBusy || state === "recording"}><span>{isBusy ? "Processing" : "Ask"}</span>{isBusy ? <Activity size={17} className="spin" /> : <Send size={17} />}</button>
              </div>
              {state === "recording" && <div className="recording-note"><span className="live-dot" /> Recording in progress. Tap stop when you're finished.</div>}
              {isBusy && <div className="processing-line"><span /> <b>Retrieving knowledge and generating answer</b></div>}
            </div>
            <div style={{ marginTop: "0.75rem" }}>
              {canPlay && audioUrlRef.current && (
                <button onClick={() => { const a = new Audio(audioUrlRef.current!); a.play().catch(() => {}); }} style={{ fontSize: "0.7rem", padding: "4px 10px", background: "rgba(120,200,120,0.15)", border: "1px solid rgba(120,200,120,0.3)", borderRadius: "4px", color: "#8c8", cursor: "pointer", marginRight: "6px" }}>
                  Play last recording
                </button>
              )}
              <button onClick={runDiag} disabled={diagBusy || !lastBlobRef.current} style={{ fontSize: "0.7rem", padding: "4px 10px", background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "4px", color: "var(--fg-secondary)", cursor: lastBlobRef.current ? "pointer" : "not-allowed" }}>
                {diagBusy ? "Running ..." : "Diagnose last recording (WebM vs OGG)"}
              </button>
              {diagResult && <pre style={{ marginTop: "0.5rem", fontSize: "0.65rem", maxHeight: "300px", overflow: "auto", padding: "8px", background: "rgba(0,0,0,0.3)", borderRadius: "4px", whiteSpace: "pre-wrap", wordBreak: "break-all", color: "var(--fg-secondary)" }}>{diagResult}</pre>}
            </div>

            {/* MIC TEST PANEL */}
            <div style={{ marginTop: "1rem", padding: "10px", border: "1px solid rgba(255,100,100,0.3)", borderRadius: "6px", background: "rgba(255,100,100,0.05)" }}>
              <div style={{ fontSize: "0.75rem", fontWeight: 600, color: "#f88", marginBottom: "6px" }}>MIC TEST (diagnostic)</div>
              <div style={{ fontSize: "0.6rem", color: "#f88", marginBottom: "6px", opacity: 0.7 }}>
                Check: chrome://settings/content/microphone and chrome://settings/media-engagement
              </div>
              <div style={{ display: "flex", gap: "6px", marginBottom: "8px", alignItems: "center" }}>
                {!micTestStream ? (
                  <button onClick={testMic} style={{ fontSize: "0.7rem", padding: "4px 10px", background: "rgba(255,100,100,0.15)", border: "1px solid rgba(255,100,100,0.3)", borderRadius: "4px", color: "#f88", cursor: "pointer" }}>
                    Test Microphone
                  </button>
                ) : (
                  <button onClick={stopMicTest} style={{ fontSize: "0.7rem", padding: "4px 10px", background: "rgba(100,200,100,0.15)", border: "1px solid rgba(100,200,100,0.3)", borderRadius: "4px", color: "#8c8", cursor: "pointer" }}>
                    Stop Mic Test
                  </button>
                )}
                {micTestStream && (
                  <span style={{ fontSize: "0.7rem", fontFamily: "monospace", color: micRms > 0.01 ? "#8c8" : "#f88" }}>
                    MIC SIGNAL RMS: {micRms.toFixed(4)} {micRms > 0.01 ? "(ACTIVE)" : "(SILENT)"}
                  </span>
                )}
              </div>
              {micTestInfo.length > 0 && (
                <div style={{ fontSize: "0.6rem", fontFamily: "monospace", lineHeight: 1.5, color: "var(--fg-secondary)", whiteSpace: "pre-wrap", maxHeight: "250px", overflow: "auto" }}>
                  {micTestInfo.join("\n")}
                </div>
              )}
              <audio ref={micTestMonitorRef} style={{ display: "block", marginTop: "6px", width: "100%", height: "32px" }} controls />
            </div>

            <div className="examples"><div className="section-label"><Sparkles size={14} /> TRY AN EXAMPLE</div>{examples.map((example) => <button key={example} onClick={() => { setQuery(example); setState("idle"); }}>{example}<ArrowUpRight size={14} /></button>)}</div>
          </section>

          <section className={`answer-column ${result ? "has-result" : ""}`}>
            {(() => {
              const genError = result ? hasGenerationError(result.latency) : false;
              const statusText = state === "error" ? "ERROR" : genError ? "GENERATION ERROR" : result ? "RESPONSE READY" : "AWAITING QUESTION";
              return <div className="answer-topline"><span className="eyebrow"><span className="eyebrow-line" /> OUTPUT / 02</span><span className="answer-count">{statusText}</span></div>;
            })()}
            {!result && state !== "error" && <div className="empty-answer"><div className="empty-art"><Radio size={25} /><span /></div><span className="empty-index">NO QUERY IN FOCUS</span><p>The answer canvas is ready.</p><small>Use the microphone for a hands-free query,<br />or send a question from the input rail.</small></div>}
            {state === "processing" && !result && <div className="loading-answer"><div className="loading-bars"><i /><i /><i /><i /><i /></div><p>Searching the knowledge base and preparing a grounded response</p><span>Frontend status - backend pipeline is processing</span></div>}
            {state === "error" && <div className="error-answer"><div className="error-icon"><X size={20} /></div><span className="empty-index">COULD NOT COMPLETE</span><h2>Let's try that again.</h2><p>{error}</p><button onClick={reset}><RotateCcw size={15} /> Reset the question</button></div>}
            {result && (() => {
              const genError = hasGenerationError(result.latency);
              return (
                <div className="result-card fade-in">
                  {result.transcript && <div className="transcript-block"><div className="result-label"><Volume2 size={14} /> TRANSCRIPT</div><p>"{result.transcript}"</p><div className="result-meta"><span><Languages size={13} /> {result.detected_language || "Language not reported"}</span></div></div>}
                  <div className="answer-block">
                    <div className="result-label"><span className={`grounded-dot ${result.grounded ? "good" : "muted"}`} /> ANSWER</div>
                    {genError ? (
                      <>
                        <div className="grounding-badge error"><CircleHelp size={14} /> Generation error</div>
                        <div className="answer-text"><span>{result.latency.error?.replace(/\b[A-Za-z0-9_-]{20,}\b/g, "[REDACTED]") || "The knowledge service encountered an error while generating the answer."}</span></div>
                      </>
                    ) : result.answer ? (
                      <>
                        <div className={`grounding-badge ${result.grounded ? "good" : "muted"}`}>{result.grounded ? <><Check size={14} /> Grounded in retrieved context</> : <><CircleHelp size={14} /> No relevant evidence found</>}</div>
                        <div className="answer-text">{result.answer}</div>
                      </>
                    ) : (
                      <>
                        <div className="grounding-badge muted"><CircleHelp size={14} /> No relevant evidence found</div>
                        <div className="answer-text"><strong>No relevant evidence found.</strong><br /><span>Try asking something covered by the knowledge base.</span></div>
                      </>
                    )}
                  </div>
                  <div className="result-footer"><span>REQUEST <b>{result.request_id.slice(0, 8)}</b></span><LatencyDetails latency={result.latency} /><button className="new-query" onClick={reset}>New question <MicOff size={14} /></button></div>
                </div>
              );
            })()}
          </section>
        </div>
        <footer className="footer"><span>HH GOA 2026 / TASK 02</span><span>RETRIEVAL - GENERATION - GROUNDING</span><span>Built for clear answers</span></footer>
      </section>
    </main>
  );
}
