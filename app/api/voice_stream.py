"""
WebSocket proxy for streaming voice STT via Sarvam Realtime API.

Browser captures PCM audio → sends over WebSocket → this endpoint
relays to Sarvam's realtime WebSocket → streams partial/final transcripts
back to the browser.

Architecture:
  Browser WebSocket  <-->  /voice/stream  <-->  Sarvam realtime WebSocket

The Sarvam API key is NEVER exposed to the browser.
"""
from __future__ import annotations

import asyncio
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

# Default VAD settings for fast conversational endpointing.
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
    """Build the Sarvam realtime WebSocket URL with query parameters."""
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
    """Streaming voice endpoint: browser PCM ↔ Sarvam realtime STT.

    Protocol (browser → backend):
      - Binary messages: raw PCM audio (linear16, 16kHz mono)
      - Text messages: JSON control events
        - {"event": "start", "lang": "en"} — begin session
        - {"event": "stop"} — end audio, wait for final transcript
        - {"event": "end"} — close session
        - {"event": "ping"} — keepalive

    Protocol (backend → browser):
      - Text messages: JSON events
        - {"event": "session.begin", "config": {...}}
        - {"event": "vad.speech_start"}
        - {"event": "vad.speech_end"}
        - {"event": "transcript.partial", "text": "...", "is_stream_final": false}
        - {"event": "transcript.final", "text": "...", "language": "en", ...}
        - {"event": "session.end", "audio_duration_s": ...}
        - {"event": "error", "message": "...", "fatal": false}
        - {"event": "latency", ...}
    """
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

    try:
        # Wait for the browser to send a 'start' event with language preference.
        start_msg = await asyncio.wait_for(ws.receive_json(), timeout=10.0)
        if start_msg.get("event") != "start":
            await ws.send_json({"event": "error", "message": "Expected start event", "fatal": True})
            await ws.close()
            return

        lang = start_msg.get("lang", "en")
        # Map ISO-639-1 to BCP-47 for Sarvam.
        bcp47_map = {
            "en": "en-IN", "hi": "hi-IN", "bn": "bn-IN", "ta": "ta-IN",
            "te": "te-IN", "mr": "mr-IN", "gu": "gu-IN", "kn": "kn-IN",
            "ml": "ml-IN", "pa": "pa-IN", "or": "or-IN", "as": "as-IN",
            "ne": "ne-IN", "ur": "ur-IN",
        }
        language_code = bcp47_map.get(lang, "en-IN")

        sarvam_url = _build_sarvam_ws_url(language_code=language_code)
        logger.info("voice_stream_connecting", language_code=language_code, url=sarvam_url)

        t_ws_connect = time.perf_counter()

        # Connect to Sarvam realtime WebSocket.
        sarvam_ws = await websockets.asyncio.client.connect(
            sarvam_url,
            additional_headers={"API-SUBSCRIPTION-KEY": api_key},
            open_timeout=10.0,
        )

        latency_ws_connect = (time.perf_counter() - t_ws_connect) * 1000
        logger.info("voice_stream_connected", latency_ms=round(latency_ws_connect, 1))

        # Send latency info back to browser.
        await ws.send_json({
            "event": "connected",
            "latency_ws_connect_ms": round(latency_ws_connect, 1),
        })

        # Run two tasks concurrently:
        # 1. Forward audio from browser → Sarvam
        # 2. Forward transcripts from Sarvam → browser
        async def forward_audio() -> None:
            """Read PCM audio from browser and send to Sarvam."""
            nonlocal audio_bytes_sent
            try:
                while True:
                    msg = await ws.receive()
                    if msg["type"] == "websocket.receive":
                        if "bytes" in msg and msg["bytes"]:
                            # Binary message: raw PCM audio.
                            import base64
                            b64_audio = base64.b64encode(msg["bytes"]).decode("ascii")
                            audio_bytes_sent += len(msg["bytes"])

                            await sarvam_ws.send(json.dumps({
                                "event": "audio_input",
                                "audio": b64_audio,
                            }))
                        elif "text" in msg and msg["text"]:
                            data = json.loads(msg["text"])
                            event = data.get("event", "")

                            if event == "stop":
                                # Send end to Sarvam to finalize.
                                await sarvam_ws.send(json.dumps({"event": "end"}))
                                logger.info("voice_stream_stop", audio_bytes=audio_bytes_sent)
                            elif event == "end":
                                # Browser wants to close.
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
            try:
                async for raw_msg in sarvam_ws:
                    if isinstance(raw_msg, str):
                        data = json.loads(raw_msg)
                        event = data.get("event", "")

                        # Forward all events to the browser.
                        await ws.send_json(data)

                        if event == "session.end":
                            total_ms = (time.perf_counter() - t_session_start) * 1000
                            await ws.send_json({
                                "event": "latency",
                                "total_session_ms": round(total_ms, 1),
                                "audio_bytes_sent": audio_bytes_sent,
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
        logger.warning("voice_stream_timeout")
        with contextlib.suppress(Exception):
            await ws.send_json({"event": "error", "message": "Session timeout", "fatal": True})
    except WebSocketDisconnect:
        logger.info("voice_stream_disconnect")
    except Exception as e:
        logger.error("voice_stream_error", error=str(e))
        with contextlib.suppress(Exception):
            await ws.send_json({"event": "error", "message": str(e), "fatal": False})
    finally:
        if sarvam_ws:
            with contextlib.suppress(Exception):
                await sarvam_ws.close()
        total_ms = (time.perf_counter() - t_session_start) * 1000
        logger.info(
            "voice_stream_closed",
            total_ms=round(total_ms, 1),
            audio_bytes=audio_bytes_sent,
        )
