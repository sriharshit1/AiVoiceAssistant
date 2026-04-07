"""
assistant.py — Core pipeline: listen → transcribe → LLM (tools + memory) → speak
Upgraded with: web search, document Q&A, code execution, weather, time, tool routing.
"""

from __future__ import annotations

import asyncio
import io
import os
import re
import subprocess
import tempfile
import time
from typing import Callable

import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
from groq import Groq
from rich.console import Console

import config
from memory import ShortTermMemory
from tools import ToolRegistry

console = Console()

CLR_USER    = "bold cyan"
CLR_AURA    = "bold magenta"
CLR_ARIA    = "bold magenta"   # alias — fixes NameError in Speaker.speak
CLR_INFO    = "dim white"
CLR_WARN    = "yellow"
CLR_ERROR   = "bold red"
CLR_SUCCESS = "bold green"
CLR_TOOL    = "bold yellow"

# ── Whisper hallucination filter ──────────────────────────
_HAL_RE = [re.compile(p, re.IGNORECASE) for p in [
    r"^(the\s+){3,}",
    r"^(thank you\.?\s*)+$",
    r"^\s*\.\s*$",
    r"^(you\.?\s*){2,}$",
    r"^(you\s+){2,}",
    r"^\s*(uh+|um+|ah+|oh+)\s*$",
    r"^[\s\W]+$",
    r"^(bye\.?\s*){2,}$",
    r"^\s*(yes+|no+|ok+|okay+)\s*$",
    r"^(\w+\s*\.?\s*){1}$",
]]

_SINGLE_WORD_BLOCKLIST = {
    "you", "the", "a", "i", "oh", "uh", "um", "ah",
    "yes", "no", "ok", "okay", "bye", "hi", "hey",
    "so", "and", "but", "or", "in", "on", "at",
}

def _is_hallucination(text: str) -> bool:
    stripped = text.strip().rstrip(".,!?").lower()
    if not stripped or len(stripped) < 3:
        return True
    if stripped in _SINGLE_WORD_BLOCKLIST:
        return True
    for p in _HAL_RE:
        if p.search(text.strip()):
            return True
    words = text.lower().split()
    if len(words) >= 4:
        if max(words.count(w) for w in set(words)) / len(words) > 0.6:
            return True
    return False


# ══════════════════════════════════════════════════════════
#  Text-to-Speech — edge-tts (Microsoft Neural)
# ══════════════════════════════════════════════════════════
class Speaker:
    def __init__(self) -> None:
        self._edge_ok  = self._check_edge()
        self._fallback = None
        self._engine   = None
        if self._edge_ok:
            console.print(f"[{CLR_SUCCESS}]✅ TTS: edge-tts ({config.EDGE_TTS_VOICE})[/]")
        else:
            self._init_fallback()

    @staticmethod
    def _check_edge() -> bool:
        try:
            import edge_tts  # noqa: F401
            return True
        except ImportError:
            return False

    def _init_fallback(self) -> None:
        for cmd in ("espeak-ng", "espeak", "say"):
            if subprocess.call(["which", cmd],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL) == 0:
                self._fallback = cmd
                console.print(f"[{CLR_INFO}]🔈 TTS fallback: {cmd}[/]")
                return
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate",   config.TTS_RATE)
            self._engine.setProperty("volume", config.TTS_VOLUME)
        except Exception as exc:
            console.print(f"[{CLR_ERROR}]No TTS backend: {exc}[/]")

    def speak(self, text: str, on_word: Callable[[str], None] | None = None) -> None:
        clean = re.sub(r"<tool:[^>]+>.*?</tool:[^>]+>", "", text, flags=re.DOTALL).strip()
        clean = re.sub(r"\s+", " ", clean)
        if not clean:
            return
        console.print(f"\n[{CLR_ARIA}]🔊 ARIA:[/] {clean}")
        try:
            if self._edge_ok:
                self._speak_edge(clean)
            elif self._fallback:
                self._speak_subprocess(clean)
            elif self._engine:
                self._engine.say(clean)
                self._engine.runAndWait()
        except Exception as exc:
            console.print(f"[{CLR_WARN}]TTS error: {exc}[/]")
        time.sleep(0.4)

    def _speak_edge(self, text: str) -> None:
        import edge_tts

        async def _synth() -> bytes:
            communicate = edge_tts.Communicate(
                text, config.EDGE_TTS_VOICE,
                rate=config.EDGE_TTS_RATE,
                volume=config.EDGE_TTS_VOLUME,
            )
            buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            return buf.getvalue()

        try:
            loop = asyncio.new_event_loop()
            try:
                data = loop.run_until_complete(_synth())
            finally:
                loop.close()

            if not data:
                raise RuntimeError("edge-tts returned empty audio")

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(data)
                tmp = f.name
            audio, sr = sf.read(tmp, dtype="float32")
            os.unlink(tmp)
            sd.play(audio, samplerate=sr)
            sd.wait()
        except Exception as exc:
            console.print(f"[{CLR_WARN}]edge-tts failed ({exc}), falling back.[/]")
            self._edge_ok = False
            self._init_fallback()
            if self._fallback:
                self._speak_subprocess(text)

    def _speak_subprocess(self, text: str) -> None:
        try:
            if self._fallback in ("espeak-ng", "espeak"):
                subprocess.run(
                    [self._fallback, "-s", str(config.TTS_RATE),
                     "-a", str(int(config.TTS_VOLUME * 200)), text],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            elif self._fallback == "say":
                subprocess.run(["say", text], check=True)
        except Exception as exc:
            console.print(f"[{CLR_WARN}]Subprocess TTS error: {exc}[/]")


# ══════════════════════════════════════════════════════════
#  Speech-to-Text (Whisper local)
# ══════════════════════════════════════════════════════════
class Listener:
    MAX_RECORD_SECONDS = 30
    MIN_RMS = 0.01

    def __init__(self) -> None:
        console.print(f"[{CLR_INFO}]⏳ Loading Whisper [{config.WHISPER_MODEL}]…[/]")
        self._model = whisper.load_model(config.WHISPER_MODEL)
        console.print(f"[{CLR_SUCCESS}]✅ Whisper ready.[/]")

    def record(self) -> np.ndarray | None:
        sr         = config.SAMPLE_RATE
        chunk_size = int(sr * 0.1)
        max_chunks = int(self.MAX_RECORD_SECONDS / 0.1)
        max_silent = int(config.SILENCE_DURATION / 0.1)
        frames, silent, spoken = [], 0, False

        console.print(f"\n[{CLR_INFO}]🎙️  Listening…[/]")
        try:
            with sd.InputStream(samplerate=sr, channels=1, dtype="int16",
                                blocksize=chunk_size, latency="low") as stream:
                for _ in range(max_chunks):
                    data, overflow = stream.read(chunk_size)
                    if overflow:
                        continue
                    frames.append(data.copy())
                    amp = int(np.abs(data).mean())
                    if amp >= config.SILENCE_THRESHOLD:
                        spoken, silent = True, 0
                    elif spoken:
                        silent += 1
                        if silent >= max_silent:
                            break
        except sd.PortAudioError as exc:
            console.print(f"[{CLR_ERROR}]Audio error: {exc}[/]")
            return None

        if not frames or not spoken:
            return None
        audio = np.concatenate(frames).flatten().astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < self.MIN_RMS:
            return None
        return audio

    def transcribe(self, audio: np.ndarray) -> str | None:
        result = self._model.transcribe(
            audio,
            fp16=False,
            language="en",
            condition_on_previous_text=False,
            no_speech_threshold=0.8,
            logprob_threshold=-2.0,
            initial_prompt="The user is asking a question or having a conversation.",
        )
        text = result["text"].strip()
        if _is_hallucination(text):
            console.print(f"[{CLR_INFO}]⚠️  Hallucination discarded: '{text[:60]}'[/]")
            return None
        return text

    def listen_and_transcribe(self) -> str | None:
        audio = self.record()
        if audio is None:
            return None
        console.print(f"[{CLR_INFO}]🧠 Transcribing…[/]")
        text = self.transcribe(audio)
        if not text:
            return None
        console.print(f"[{CLR_USER}]🗣️  You:[/] {text}")
        return text


# ══════════════════════════════════════════════════════════
#  Brain — Groq LLM + Memory + Tool Routing
# ══════════════════════════════════════════════════════════
class Brain:
    _TOOL_RE = re.compile(
        r"<tool:(?P<name>\w+)>(?P<body>.*?)</tool:\w+>",
        re.DOTALL,
    )

    def __init__(self) -> None:
        self._client  = Groq(api_key=config.GROQ_API_KEY)
        self.tools    = ToolRegistry()
        self.memory   = ShortTermMemory(
            max_turns=config.MEMORY_MAX_TURNS,
            summary_every=config.MEMORY_SUMMARY_EVERY,
        )
        p = config.PROMPTS["system"]
        self._base_system = "\n\n".join([
            p["persona"].strip(),
            p["fallback"].strip(),
            p["capabilities"].strip(),
            self.tools.describe(),
        ])

    def _build_system_with_context(self) -> str:
        """Inject live date/time + user location into system prompt every request."""
        import datetime
        try:
            import zoneinfo
            tz = zoneinfo.ZoneInfo(config.USER_TIMEZONE)
        except Exception:
            tz = None
        now = datetime.datetime.now(tz=tz)
        date_str = now.strftime("%A, %d %B %Y")
        time_str = now.strftime("%I:%M %p")
        tz_str   = config.USER_TIMEZONE

        context_block = (
            f"LIVE CONTEXT (always accurate — use naturally in replies):\n"
            f"  Current date : {date_str}\n"
            f"  Current time : {time_str} ({tz_str})\n"
            f"  User location: {config.DEFAULT_CITY}, {config.DEFAULT_COUNTRY}"
        )
        return self._base_system + "\n\n" + context_block

    def chat(
        self,
        user_text: str,
        status_cb: Callable[[str], None] | None = None,
    ) -> str:
        self.memory.add("user", user_text)
        system_prompt = self._build_system_with_context()
        messages = self.memory.build_messages(system_prompt)

        if status_cb:
            status_cb("thinking")

        raw = self._call_llm(messages)
        if raw is None:
            return config.PROMPTS["error_message"]

        tool_results: list[str] = []
        for match in self._TOOL_RE.finditer(raw):
            name = match.group("name")
            body = match.group("body").strip()
            console.print(f"[{CLR_TOOL}]🔧 Tool: {name}[/]")
            if status_cb:
                status_cb(f"using tool: {name}")
            result = self._execute_tool(name, body)
            tool_results.append(f"<tool_result:{name}>\n{result}\n</tool_result:{name}>")

        if tool_results:
            tool_context = "\n\n".join(tool_results)
            follow_up_messages = messages + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        f"Tool results:\n{tool_context}\n\n"
                        "Now give your final answer to the user based on these results. "
                        "Be concise and natural for voice."
                    ),
                },
            ]
            if status_cb:
                status_cb("processing results")
            final = self._call_llm(follow_up_messages)
            reply = final if final else raw
        else:
            reply = raw

        reply = self._TOOL_RE.sub("", reply).strip()
        self.memory.add("assistant", reply)
        return reply

    def _call_llm(self, messages: list[dict]) -> str | None:
        try:
            resp = self._client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=messages,
                temperature=config.GROQ_TEMPERATURE,
                max_tokens=config.GROQ_MAX_TOKENS,
                top_p=config.GROQ_TOP_P,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            console.print(f"[{CLR_ERROR}]Groq error: {exc}[/]")
            return None

    def _execute_tool(self, name: str, body: str) -> str:
        if name == "search":
            return self.tools.search.search(body)
        elif name == "doc":
            return self.tools.docs.query(body)
        elif name == "code":
            code = self.tools.code.extract_code(body) or body
            return self.tools.code.run(code)
        elif name == "weather":
            return self.tools.weather.get_weather(body)
        elif name == "time":
            return self.tools.time.get_time(body)
        return f"Unknown tool: {name}"

    def reset(self) -> None:
        self.memory.reset()


# ══════════════════════════════════════════════════════════
#  VoiceAssistant — CLI orchestrator
# ══════════════════════════════════════════════════════════
class VoiceAssistant:
    def __init__(self) -> None:
        self.speaker  = Speaker()
        self.listener = Listener()
        self.brain    = Brain()
        self._exit_phrases = [
            p.lower()
            for p in config.PROMPTS.get("exit_phrases", ["goodbye", "exit"])
        ]

    def _is_exit(self, text: str) -> bool:
        return any(phrase in text.lower() for phrase in self._exit_phrases)

    def _print_banner(self) -> None:
        console.rule("[bold magenta]✨ ARIA Voice Assistant ✨[/]")
        console.print(
            f"  Model   : [{CLR_SUCCESS}]{config.GROQ_MODEL}[/]\n"
            f"  Whisper : [{CLR_SUCCESS}]{config.WHISPER_MODEL}[/]\n"
            f"  Voice   : [{CLR_SUCCESS}]{config.EDGE_TTS_VOICE}[/]\n"
            f"  Memory  : [{CLR_SUCCESS}]{config.MEMORY_MAX_TURNS} turns[/]\n"
            f"  Tools   : [{CLR_SUCCESS}]web search · doc Q&A · code exec · weather · time[/]\n"
            f"  City    : [{CLR_SUCCESS}]{config.DEFAULT_CITY}, {config.DEFAULT_COUNTRY}[/]\n"
        )
        console.rule()

    def run(self) -> None:
        self._print_banner()
        self.speaker.speak(config.PROMPTS["startup_message"])

        while True:
            try:
                user_text = self.listener.listen_and_transcribe()
                if not user_text:
                    self.speaker.speak(config.PROMPTS["mishear_message"])
                    continue
                if self._is_exit(user_text):
                    self.speaker.speak(config.PROMPTS["goodbye_message"])
                    break
                reply = self.brain.chat(user_text)
                self.speaker.speak(reply)
            except KeyboardInterrupt:
                self.speaker.speak(config.PROMPTS["goodbye_message"])
                break
            except Exception as exc:
                console.print(f"[{CLR_ERROR}]Error: {exc}[/]")
                self.speaker.speak(config.PROMPTS["error_message"])