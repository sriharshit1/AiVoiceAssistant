"""
server.py — FastAPI backend for AURA Web UI
Fix: replaced all asyncio.run() calls inside async routes with
     proper await / loop.run_in_executor patterns.
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import subprocess
import tempfile
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import (
    FastAPI, File, HTTPException,
    UploadFile, WebSocket, WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from assistant import Brain, Listener, Speaker



# ── App ────────────────────────────────────────────────────
app = FastAPI(title="AURA Voice Assistant", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

STATIC_DIR  = Path(__file__).parent / "static"
UPLOADS_DIR = Path(__file__).parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Singletons ─────────────────────────────────────────────
_brain:    Brain    | None = None
_speaker:  Speaker  | None = None
_listener: Listener | None = None


# ── WebSocket manager ──────────────────────────────────────
class ConnectionManager:
    def __init__(self) -> None:
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict) -> None:
        for ws in list(self.active):
            try:
                await ws.send_json(data)
            except Exception:
                self.disconnect(ws)

manager = ConnectionManager()


@app.on_event("startup")
async def startup() -> None:
    global _brain, _speaker, _listener
    loop = asyncio.get_running_loop()
    _brain    = await loop.run_in_executor(None, Brain)
    _speaker  = await loop.run_in_executor(None, Speaker)
    _listener = await loop.run_in_executor(None, Listener)


# ── WebSocket ──────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)


# ── Pydantic models ────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    speak: bool = True


# ── Routes ────────────────────────────────────────────────
@app.get("/")
async def root() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/api/chat")
async def chat(req: ChatRequest) -> JSONResponse:
    if not _brain:
        raise HTTPException(503, "Brain not initialised yet, please wait a moment.")

    loop = asyncio.get_running_loop()

    # Status callback — safe to call from a thread-pool worker
    def status_cb(msg: str) -> None:
        asyncio.run_coroutine_threadsafe(
            manager.broadcast({"type": "status", "msg": msg}), loop
        )

    # Run blocking brain.chat in thread pool
    try:
        reply = await loop.run_in_executor(
            None, lambda: _brain.chat(req.message, status_cb)
        )
    except Exception as exc:
        import traceback; traceback.print_exc()
        raise HTTPException(500, str(exc))

    # TTS — run async edge-tts properly in the event loop
    audio_b64 = None
    if req.speak:
        try:
            audio_b64 = await _tts_to_base64_async(reply)
        except Exception:
            audio_b64 = None

    await manager.broadcast({"type": "status", "msg": "done"})

    return JSONResponse({
        "reply":     reply,
        "audio_b64": audio_b64,
        "memory":    _get_memory_dict(),
    })


@app.post("/api/voice")
async def voice(audio: UploadFile = File(...)) -> JSONResponse:
    if not (_brain and _listener):
        raise HTTPException(503, "Services not initialised yet.")

    suffix = Path(audio.filename or "audio.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(await audio.read())
        tmp_path = f.name

    loop = asyncio.get_running_loop()

    try:
        audio_array = await loop.run_in_executor(None, lambda: _load_audio(tmp_path))
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    if audio_array is None:
        return JSONResponse({"error": "Could not decode audio — ensure ffmpeg is installed (sudo apt install ffmpeg)"}, status_code=400)

    # Transcribe
    await manager.broadcast({"type": "status", "msg": "transcribing"})
    transcript = await loop.run_in_executor(
        None, lambda: _listener.transcribe(audio_array)
    )
    if not transcript:
        return JSONResponse({"error": "Speech not detected. Speak clearly for at least 2 seconds after clicking the mic button."}, status_code=422)

    # Chat
    await manager.broadcast({"type": "status", "msg": "thinking"})

    def status_cb(msg: str) -> None:
        asyncio.run_coroutine_threadsafe(
            manager.broadcast({"type": "status", "msg": msg}), loop
        )

    try:
        reply = await loop.run_in_executor(
            None, lambda: _brain.chat(transcript, status_cb)
        )
    except Exception as exc:
        import traceback; traceback.print_exc()
        raise HTTPException(500, str(exc))

    # TTS
    await manager.broadcast({"type": "status", "msg": "speaking"})
    try:
        audio_b64 = await _tts_to_base64_async(reply)
    except Exception:
        audio_b64 = None

    await manager.broadcast({"type": "status", "msg": "done"})

    return JSONResponse({
        "transcript": transcript,
        "reply":      reply,
        "audio_b64":  audio_b64,
        "memory":     _get_memory_dict(),
    })


@app.post("/api/upload")
async def upload_doc(file: UploadFile = File(...)) -> JSONResponse:
    if not _brain:
        raise HTTPException(503, "Brain not initialised yet.")
    dest = UPLOADS_DIR / (file.filename or f"upload_{uuid.uuid4().hex[:8]}")
    dest.write_bytes(await file.read())
    loop   = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, lambda: _brain.tools.docs.ingest(str(dest))
    )
    return JSONResponse({"result": result, "docs": _brain.tools.docs.list_docs()})


@app.get("/api/docs")
async def list_docs() -> JSONResponse:
    if not _brain:
        raise HTTPException(503, "Brain not initialised yet.")
    return JSONResponse({"docs": _brain.tools.docs.list_docs()})


@app.delete("/api/docs/{filename}")
async def remove_doc(filename: str) -> JSONResponse:
    if not _brain:
        raise HTTPException(503, "Brain not initialised yet.")
    result = _brain.tools.docs.remove(filename)
    return JSONResponse({"result": result, "docs": _brain.tools.docs.list_docs()})


@app.post("/api/reset")
async def reset_memory() -> JSONResponse:
    if _brain:
        _brain.reset()
    return JSONResponse({"status": "memory cleared"})


@app.get("/api/status")
async def status() -> JSONResponse:
    if not _brain:
        return JSONResponse({"status": "initialising"})
    return JSONResponse({
        "status":  "ok",
        "model":   config.GROQ_MODEL,
        "whisper": config.WHISPER_MODEL,
        "voice":   config.EDGE_TTS_VOICE,
        "memory":  _get_memory_dict(),
    })


# ── Helpers ───────────────────────────────────────────────

def _get_memory_dict() -> dict:
    if not _brain:
        return {}
    return {
        "turns":    _brain.memory.turn_count(),
        "duration": _brain.memory.session_duration(),
        "facts":    _brain.memory.all_facts(),
        "docs":     _brain.tools.docs.list_docs(),
    }


async def _tts_to_base64_async(text: str) -> str | None:
    """
    Proper async edge-tts synthesis — no asyncio.run() inside event loop.
    Streams audio chunks directly into a BytesIO buffer.
    """
    try:
        import edge_tts
        communicate = edge_tts.Communicate(
            text,
            config.EDGE_TTS_VOICE,
            rate=config.EDGE_TTS_RATE,
            volume=config.EDGE_TTS_VOLUME,
        )
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        data = buf.getvalue()
        if not data:
            return None
        return base64.b64encode(data).decode()
    except Exception as exc:
        print(f"[TTS] edge-tts error: {exc}")
        return None


def _load_audio(path: str) -> np.ndarray | None:
    """Load any audio file to 16kHz mono float32 via ffmpeg."""
    wav_path = path + "_converted.wav"
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-ar", "16000", "-ac", "1", wav_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"[Audio] ffmpeg error:\n{result.stderr[-800:]}")
            return None
        audio, _ = sf.read(wav_path, dtype="float32")
        return audio
    except FileNotFoundError:
        print("[Audio] ffmpeg not found — install with: sudo apt install ffmpeg")
        return None
    except Exception as exc:
        print(f"[Audio] decode error: {exc}")
        return None
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        reload=False,
        log_level="info",
    )