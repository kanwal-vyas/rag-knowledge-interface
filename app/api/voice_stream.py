"""
WebSocket proxy for streaming voice STT via Sarvam Realtime API.

With comprehensive latency instrumentation at every boundary.
All timestamps are in milliseconds using time.perf_counter().
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import time
from typing import Any

import structlog
import websockets.asyncio.client
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.config import get_settings

logger = structlog.get_logger(__name__)

router = APIRouter()

_SARVAM_REALTIME_WS_URL = "wss://api.sarvam.ai/speech-to-text-realtime/ws"

_DEFAULT_VAD = {
    "threshold": "0.3",
    "silence_duration_ms": "500",
    "min_speech_duration_ms": "250",
}


def _build_sarvam_ws_url(
    *,
    language_code: str = "en-IN",
    mode: str = "transcribe",
    stream_type: str = "fast",
) -> str:
    params = {
        "language_code": language_code,
        "model": "saaras:v3-realtime",
        "stream_type": stream_type,
        "mode": mode,
        "endpointing": "vad",
        "encoding": "linear16",
        "sample_rate": "16000",
        "return_timestamps": "true",
        **_DEFAULT_VAD,
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{_SARVAM_REALTIME_WS_URL}?{query}"


@router.websocket("/voice/stream")
async def voice_stream(ws: WebSocket) -> None:
    """Streaming voice endpoint with full latency instrumentation."""
    client = ws.client
    logger.info(
        "voice_stream_accepted",
        client_host=client.host if client else None,
        client_port=client.port if client else None,
    )
    await ws.accept()
    settings = get_settings()
    api_key = settings.stt_api_key

    if not api_key:
        await ws.send_json({
            "event": "error",
            "message": "SARVAM_API_KEY not configured",
            "fatal": True,
        })
        await ws.close()
        return

    sarvam_ws: Any = None
    t_session_start = time.perf_counter()
    audio_bytes_sent = 0
    audio_chunks_forwarded = 0
    first_chunk_time: float | None = None
    first_partial_time: float | None = None
    first_partial_text = ""
    final_transcript_time: float | None = None
    final_transcript_text = ""

    def ts_ms(label: str) -> float:
        """Log a timestamp relative to session start."""
        ms = (time.perf_counter() - t_session_start) * 1000
        logger.info("voice_stream_ts", session_id=session_id, label=label, ms=round(ms, 1))
        return ms

    session_id = ""

    try:
        # Wait for the browser to send a 'start' event.
        start_msg = await asyncio.wait_for(ws.receive_json(), timeout=10.0)
        if start_msg.get("event") != "start":
            await ws.send_json({"event": "error", "message": "Expected start event", "fatal": True})
            await ws.close()
            return

        session_id = start_msg.get("session_id", "unknown")
        lang = start_msg.get("lang", "en")

        ts_ms("B1: start event received from browser")

        # Map ISO-639-1 to BCP-47.
        bcp47_map = {
            "en": "en-IN", "hi": "hi-IN", "bn": "bn-IN", "ta": "ta-IN",
            "te": "te-IN", "mr": "mr-IN", "gu": "gu-IN", "kn": "kn-IN",
            "ml": "ml-IN", "pa": "pa-IN", "or": "or-IN", "as": "as-IN",
            "ne": "ne-IN", "ur": "ur-IN",
        }
        language_code = bcp47_map.get(lang, "en-IN")

        sarvam_url = _build_sarvam_ws_url(language_code=language_code)
        ts_ms("B2: Sarvam URL built")

        # Connect to Sarvam realtime WebSocket.
        t_ws_connect = time.perf_counter()
        try:
            sarvam_ws = await websockets.asyncio.client.connect(
                sarvam_url,
                additional_headers={"API-SUBSCRIPTION-KEY": api_key},
                open_timeout=10.0,
            )
            latency_ws_connect = (time.perf_counter() - t_ws_connect) * 1000
            ts_ms(f"B3: Sarvam WS connected ({latency_ws_connect:.0f}ms)")
        except Exception as sarvam_err:
            latency_ws_connect = (time.perf_counter() - t_ws_connect) * 1000
            logger.error(
                "voice_stream_sarvam_connect_failed",
                session_id=session_id,
                error_type=type(sarvam_err).__name__,
                error=str(sarvam_err),
                latency_ms=round(latency_ws_connect, 1),
                url=sarvam_url,
            )
            with contextlib.suppress(Exception):
                await ws.send_json({
                    "event": "error",
                    "message": f"Failed to connect to Sarvam STT: {sarvam_err}",
                    "fatal": True,
                })
            await ws.close()
            return

        # Send connected event back to browser.
        await ws.send_json({
            "event": "connected",
            "latency_ws_connect_ms": round(latency_ws_connect, 1),
            "session_id": session_id,
        })

        async def forward_audio() -> None:
            """Read PCM audio from browser and send to Sarvam."""
            nonlocal audio_bytes_sent, audio_chunks_forwarded, first_chunk_time
            try:
                while True:
                    msg = await ws.receive()
                    if msg["type"] == "websocket.receive":
                        if "bytes" in msg and msg["bytes"]:
                            pcm_data = msg["bytes"]
                            audio_bytes_sent += len(pcm_data)
                            audio_chunks_forwarded += 1

                            if first_chunk_time is None:
                                first_chunk_time = time.perf_counter()
                                ts_ms("B4: first PCM chunk from browser")

                            # Forward to Sarvam.
                            b64_audio = base64.b64encode(pcm_data).decode("ascii")
                            await sarvam_ws.send(json.dumps({
                                "event": "audio_input",
                                "audio": b64_audio,
                            }))

                            if audio_chunks_forwarded == 1:
                                ts_ms("B5: first PCM chunk forwarded to Sarvam")

                        elif "text" in msg and msg["text"]:
                            data = json.loads(msg["text"])
                            event = data.get("event", "")

                            if event == "stop":
                                ts_ms("B8: stop event received")
                                await sarvam_ws.send(json.dumps({"event": "end"}))
                            elif event == "end":
                                await sarvam_ws.send(json.dumps({"event": "end"}))
                                return
                            elif event == "ping":
                                await sarvam_ws.send(json.dumps({"event": "ping"}))
                    elif msg["type"] == "websocket.disconnect":
                        return
            except Exception as e:
                logger.error("voice_stream_forward_audio_error", error=str(e))

        async def forward_transcripts() -> None:
            """Read messages from Sarvam and forward to browser."""
            nonlocal first_partial_time, first_partial_text
            nonlocal final_transcript_time, final_transcript_text
            try:
                async for raw_msg in sarvam_ws:
                    if isinstance(raw_msg, str):
                        data = json.loads(raw_msg)
                        event = data.get("event", "")

                        if event == "transcript.partial" and first_partial_time is None:
                            first_partial_time = time.perf_counter()
                            first_partial_text = data.get("text", "")
                            ts_ms(f"B6: FIRST partial from Sarvam: \"{first_partial_text[:50]}\"")

                        if event == "transcript.final":
                            final_transcript_time = time.perf_counter()
                            final_transcript_text = data.get("text", "")
                            ts_ms("B9: FINAL transcript from Sarvam")

                        # Forward all events to the browser.
                        await ws.send_json(data)

                        if event == "session.end":
                            total_ms = (time.perf_counter() - t_session_start) * 1000

                            # Calculate breakdown.
                            breakdown: dict[str, float] = {}
                            if first_chunk_time:
                                breakdown["first_chunk_from_browser_ms"] = round(
                                    (first_chunk_time - t_session_start) * 1000, 1,
                                )
                            if first_partial_time and first_chunk_time:
                                breakdown["sarvam_first_partial_ms"] = round(
                                    (first_partial_time - first_chunk_time) * 1000, 1,
                                )
                            if final_transcript_time and first_partial_time:
                                breakdown["sarvam_finalization_ms"] = round(
                                    (final_transcript_time - first_partial_time) * 1000, 1,
                                )

                            ts_ms("B10: session.end")

                            await ws.send_json({
                                "event": "latency",
                                "total_session_ms": round(total_ms, 1),
                                "audio_bytes_sent": audio_bytes_sent,
                                "audio_chunks": audio_chunks_forwarded,
                                "backend_breakdown": breakdown,
                                "first_partial_text": first_partial_text,
                                "final_transcript_text": final_transcript_text,
                            })
                            return
                        elif event == "error" and data.get("is_fatal"):
                            return
            except Exception as e:
                logger.error("voice_stream_forward_transcript_error", error=str(e))
                with contextlib.suppress(Exception):
                    await ws.send_json({"event": "error", "message": str(e), "fatal": False})

        # Run both forwarding tasks concurrently.
        await asyncio.gather(
            forward_audio(),
            forward_transcripts(),
            return_exceptions=True,
        )

    except TimeoutError:
        logger.warning("voice_stream_timeout", session_id=session_id)
        with contextlib.suppress(Exception):
            await ws.send_json({"event": "error", "message": "Session timeout", "fatal": True})
    except WebSocketDisconnect:
        ts_ms("WebSocket disconnected")
    except Exception as e:
        logger.error("voice_stream_error", session_id=session_id, error=str(e))
        with contextlib.suppress(Exception):
            await ws.send_json({"event": "error", "message": str(e), "fatal": False})
    finally:
        if sarvam_ws:
            with contextlib.suppress(Exception):
                await sarvam_ws.close()
        total_ms = (time.perf_counter() - t_session_start) * 1000
        logger.info(
            "voice_stream_closed",
            session_id=session_id,
            total_ms=round(total_ms, 1),
            audio_bytes=audio_bytes_sent,
            audio_chunks=audio_chunks_forwarded,
        )
