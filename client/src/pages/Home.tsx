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
import { getHealth, queryText, type Latency, type QueryResponse } from "@/lib/api";
import { VoiceStreamClient } from "@/lib/voiceStream";

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
  const [health, setHealth] = useState("checking");
  const [voiceLang, setVoiceLang] = useState("en");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const streamClientRef = useRef<VoiceStreamClient | null>(null);
  const streamStopTimerRef = useRef<number | undefined>(undefined);
  const streamStoppingRef = useRef(false);
  const streamFinalReceivedRef = useRef(false);
  const [partialText, setPartialText] = useState("");
  const [streamLatency, setStreamLatency] = useState<string[]>([]);

  useEffect(() => {
    getHealth().then((data) => setHealth(data.status === "ok" ? "online" : "degraded")).catch(() => setHealth("offline"));
    return () => {
      if (streamStopTimerRef.current) window.clearTimeout(streamStopTimerRef.current);
      const streamClient = streamClientRef.current;
      streamClientRef.current = null;
      streamStoppingRef.current = false;
      streamFinalReceivedRef.current = false;
      streamClient?.stop();
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

  // ── STREAMING VOICE ──
  const startStreamingVoice = async () => {
    // The ref is updated synchronously, so rapid double-clicks cannot create
    // parallel microphone/WebSocket sessions before React rerenders.
    if (streamClientRef.current || streamStoppingRef.current || state === "recording" || state === "processing") return;
    setError("");
    setPartialText("");
    setStreamLatency([]);
    streamFinalReceivedRef.current = false;
    setState("recording");

    const client = new VoiceStreamClient({
      onConnected: (latencyMs) => {
        if (streamClientRef.current !== client) return;
        setStreamLatency(prev => [...prev, `WS connect: ${latencyMs.toFixed(0)}ms`]);
      },
      onTimestamp: (label, ms) => {
        // Frontend timestamps are logged via console in voiceStream.ts.
      },
      onPartial: (text) => {
        if (streamClientRef.current !== client || streamStoppingRef.current) return;
        setPartialText(text);
        setState("recording");
      },
      onFinal: (text, lang) => {
        if (streamClientRef.current !== client) return;
        streamStoppingRef.current = false;
        streamFinalReceivedRef.current = true;
        setPartialText("");
        setState("processing");
        // Use the final streaming transcript for the RAG query.
        queryText(text)
          .then((qr) => {
            setResult({ ...qr, transcript: text, detected_language: lang || "en" });
            setQuery(text);
            setState("complete");
          })
          .catch((err) => {
            setError(err instanceof Error ? err.message : "Generation failed");
            setState("error");
          });
      },
      onSpeechStart: () => {
        if (streamClientRef.current !== client) return;
        setStreamLatency(prev => [...prev, `Speech started`]);
      },
      onSpeechEnd: () => {
        if (streamClientRef.current !== client) return;
        setStreamLatency(prev => [...prev, `Speech ended — finalizing...`]);
      },
      onError: (msg, fatal) => {
        if (streamClientRef.current !== client) return;
        if (fatal) {
          if (streamStopTimerRef.current) window.clearTimeout(streamStopTimerRef.current);
          streamStopTimerRef.current = undefined;
          streamStoppingRef.current = false;
          streamFinalReceivedRef.current = false;
          streamClientRef.current = null;
          setError(`Streaming error: ${msg}`);
          setState("error");
          client.stop();
        } else {
          setStreamLatency(prev => [...prev, `Warning: ${msg}`]);
        }
      },
      onLatency: (totalMs, audioBytes, backendBreakdown) => {
        if (streamClientRef.current !== client) return;
        setStreamLatency(prev => {
          const next = [...prev, `--- Backend breakdown ---`];
          if (backendBreakdown) {
            for (const [k, v] of Object.entries(backendBreakdown)) {
              next.push(`  ${k}: ${v}ms`);
            }
          }
          next.push(`  total_session: ${totalMs.toFixed(0)}ms`);
          next.push(`  audio_bytes: ${audioBytes}`);
          return next;
        });
      },
      onStateChange: (nextState) => {
        if (nextState === "idle" && streamClientRef.current === client) {
          if (streamStopTimerRef.current) window.clearTimeout(streamStopTimerRef.current);
          streamStopTimerRef.current = undefined;
          streamClientRef.current = null;
          if (streamStoppingRef.current) {
            streamStoppingRef.current = false;
            streamFinalReceivedRef.current = false;
            setPartialText("");
            setState("idle");
          }
        }
      },
    });

    streamClientRef.current = client;
    await client.start(voiceLang || "en");
  };

  const stopStreamingVoice = () => {
    const client = streamClientRef.current;
    if (!client || streamStoppingRef.current) return;

    streamStoppingRef.current = true;
    client.stopRecording();
    // Local capture is already released by stopRecording(); show the normal
    // non-streaming UI while the backend has a brief chance to send its final.
    if (streamClientRef.current !== client || client.state === "idle") {
      streamStoppingRef.current = false;
      setPartialText("");
      setState("idle");
      return;
    }
    setState("processing");

    if (streamStopTimerRef.current) window.clearTimeout(streamStopTimerRef.current);
    streamStopTimerRef.current = window.setTimeout(() => {
      if (streamClientRef.current === client) {
        client.stop();
        streamClientRef.current = null;
        if (!streamFinalReceivedRef.current) {
          streamStoppingRef.current = false;
          setPartialText("");
          setState("idle");
        }
      }
      streamStopTimerRef.current = undefined;
    }, 5000);
  };

  const reset = () => { setQuery(""); setResult(null); setError(""); setState("idle"); setTimeout(() => textareaRef.current?.focus(), 0); };
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
              <div className="composer-head"><span>{state === "recording" ? "Listening" : state === "processing" ? "Working through the pipeline" : "Your question"}</span><span className="composer-mode">{state === "recording" ? "LIVE / VOICE" : "TEXT / VOICE"}</span></div>
              <textarea ref={textareaRef} value={query} onChange={(event) => { setQuery(event.target.value); if (state === "error") setState("idle"); }} onKeyDown={(event) => { if ((event.metaKey || event.ctrlKey) && event.key === "Enter") submitText(); }} placeholder="What would you like to understand?" disabled={state === "recording" || isBusy} aria-label="Ask the knowledge base" />
              <div className="composer-actions">
                <div className="voice-controls">
                  {state === "recording" ? <button className="mic-button recording" onClick={stopStreamingVoice} aria-label="Stop recording"><Square size={18} fill="currentColor" /><span className="pulse-ring" /></button> : <button className="mic-button" onClick={startStreamingVoice} disabled={isBusy} aria-label="Stream voice question"><Mic size={20} /></button>}
                  <select className="voice-lang-select" value={voiceLang} onChange={(e) => setVoiceLang(e.target.value)} disabled={isBusy} aria-label="Voice language">
                    {voiceLanguages.map((l) => <option key={l.value} value={l.value}>{l.label}</option>)}
                  </select>
                </div>
                <button className="send-button" onClick={submitText} disabled={!query.trim() || isBusy || state === "recording"}><span>{isBusy ? "Processing" : "Ask"}</span>{isBusy ? <Activity size={17} className="spin" /> : <Send size={17} />}</button>
              </div>
              {state === "recording" && (
                <>
                  {partialText && <div style={{ marginTop: "6px", padding: "8px", background: "rgba(255,255,255,0.05)", borderRadius: "4px", fontFamily: "monospace", fontSize: "0.8rem", color: "var(--fg-secondary)" }}><span style={{ color: "var(--accent)", fontWeight: 600 }}>LIVE:</span> {partialText}<span style={{ animation: "blink 1s infinite" }}>|</span></div>}
                  <div className="recording-note"><span className="live-dot" /> Streaming. Tap stop when done.</div>
                </>
              )}
              {isBusy && <div className="processing-line"><span /> <b>Retrieving knowledge and generating answer</b></div>}
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
                  <div className="result-footer"><span>REQUEST <b>{result.request_id.slice(0, 8)}</b></span><LatencyDetails latency={result.latency} />{streamLatency.length > 0 && <details className="latency-details"><summary><Clock3 size={14} /> Stream latency</summary><div className="latency-grid">{streamLatency.map((l, i) => <span key={i}><small>{l}</small></span>)}</div></details>}<button className="new-query" onClick={reset}>New question <MicOff size={14} /></button></div>
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
