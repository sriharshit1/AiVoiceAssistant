"""
tools.py — Advanced capability tools for ARIA
  1. WebSearchTool   — DuckDuckGo search (no API key needed)
  2. DocumentQATool  — Ask questions about uploaded files (PDF / TXT / MD)
  3. CodeExecTool    — Safe sandboxed Python execution
  4. WeatherTool     — Live weather via wttr.in (no API key needed)
  5. TimeTool        — Current date/time for any city/timezone
"""

from __future__ import annotations

import ast
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()

# ══════════════════════════════════════════════════════════
#  1. Web Search — DuckDuckGo (no API key required)
# ══════════════════════════════════════════════════════════
class WebSearchTool:
    """Searches DuckDuckGo and returns top N text snippets."""

    MAX_RESULTS = 5

    def search(self, query: str) -> str:
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.MAX_RESULTS))
            if not results:
                return f"No results found for: {query}"
            parts = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "")
                body  = r.get("body",  "")
                href  = r.get("href",  "")
                parts.append(f"[{i}] {title}\n{body}\nSource: {href}")
            return "\n\n".join(parts)
        except ImportError:
            return "Web search unavailable. Install: pip install duckduckgo-search"
        except Exception as exc:
            return f"Search error: {exc}"


# ══════════════════════════════════════════════════════════
#  2. Document Q&A — PDF / TXT / MD
# ══════════════════════════════════════════════════════════
class DocumentQATool:
    """
    Loads uploaded documents into a simple in-memory store.
    Returns relevant chunks using keyword overlap (no vector DB needed).
    """

    CHUNK_SIZE    = 800   # characters per chunk
    CHUNK_OVERLAP = 100
    TOP_K = 4

    def __init__(self) -> None:
        self._docs: dict[str, list[str]] = {}

    def ingest(self, file_path: str) -> str:
        path = Path(file_path)
        if not path.exists():
            return f"File not found: {file_path}"
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                text = self._read_pdf(path)
            elif suffix in (".txt", ".md", ".csv", ".py", ".js", ".ts", ".json"):
                text = path.read_text(encoding="utf-8", errors="ignore")
            else:
                return f"Unsupported file type: {suffix}"
            chunks = self._chunk(text)
            self._docs[path.name] = chunks
            return (
                f"✅ Loaded '{path.name}' — "
                f"{len(text):,} chars, {len(chunks)} chunks."
            )
        except Exception as exc:
            return f"Failed to load document: {exc}"

    def _read_pdf(self, path: Path) -> str:
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            return "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
        except ImportError:
            try:
                import pdfplumber
                with pdfplumber.open(str(path)) as pdf:
                    return "\n".join(
                        p.extract_text() or "" for p in pdf.pages
                    )
            except ImportError:
                return "[PDF reading requires: pip install pypdf OR pdfplumber]"

    def _chunk(self, text: str) -> list[str]:
        chunks, start = [], 0
        while start < len(text):
            end = start + self.CHUNK_SIZE
            chunks.append(text[start:end])
            start = end - self.CHUNK_OVERLAP
        return chunks

    def query(self, question: str, filename: str | None = None) -> str:
        if not self._docs:
            return "No documents loaded yet. Please upload a file first."
        docs_to_search = (
            {filename: self._docs[filename]}
            if filename and filename in self._docs
            else self._docs
        )
        scored: list[tuple[float, str, str]] = []
        q_words = set(re.findall(r"\w+", question.lower()))
        for fname, chunks in docs_to_search.items():
            for chunk in chunks:
                c_words = set(re.findall(r"\w+", chunk.lower()))
                overlap = len(q_words & c_words) / max(len(q_words), 1)
                scored.append((overlap, chunk, fname))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: self.TOP_K]
        if not top or top[0][0] == 0:
            return "I couldn't find relevant content in the uploaded documents."
        context_parts = [
            f"[From {fname}]\n{chunk}"
            for _, chunk, fname in top
        ]
        return "DOCUMENT CONTEXT:\n\n" + "\n\n---\n\n".join(context_parts)

    def list_docs(self) -> list[str]:
        return list(self._docs.keys())

    def remove(self, filename: str) -> str:
        if filename in self._docs:
            del self._docs[filename]
            return f"Removed '{filename}'"
        return f"'{filename}' not found"


# ══════════════════════════════════════════════════════════
#  3. Code Execution — sandboxed subprocess
# ══════════════════════════════════════════════════════════
class CodeExecTool:
    """
    Executes Python code in a subprocess with a timeout.
    Returns stdout + stderr as a string.
    """

    TIMEOUT_SECONDS  = 10
    MAX_OUTPUT_CHARS = 2000

    _BLOCKED = re.compile(
        r"\b(os\.system|subprocess|shutil\.rmtree|open\s*\(.*[\"\']/|"
        r"__import__|importlib|socket|requests|httpx|urllib)\b",
        re.IGNORECASE,
    )

    def run(self, code: str) -> str:
        if self._BLOCKED.search(code):
            return (
                "⛔ Blocked: code contains disallowed operations "
                "(filesystem writes, network calls, subprocess)."
            )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name
        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.TIMEOUT_SECONDS,
            )
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += "\nSTDERR:\n" + result.stderr
            output = output.strip()
            if not output:
                output = "(no output)"
            if len(output) > self.MAX_OUTPUT_CHARS:
                output = output[: self.MAX_OUTPUT_CHARS] + "\n… (truncated)"
            return output
        except subprocess.TimeoutExpired:
            return f"⏱️ Execution timed out after {self.TIMEOUT_SECONDS}s."
        except Exception as exc:
            return f"Execution error: {exc}"
        finally:
            os.unlink(tmp_path)

    @staticmethod
    def extract_code(text: str) -> str | None:
        match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        if any(kw in text for kw in ("def ", "import ", "print(", "=")):
            return text.strip()
        return None


# ══════════════════════════════════════════════════════════
#  4. Weather Tool — wttr.in (no API key required)
# ══════════════════════════════════════════════════════════
class WeatherTool:
    """
    Fetches current weather for any city using wttr.in JSON API.
    No API key required. Falls back gracefully on network errors.
    """

    BASE_URL = "https://wttr.in/{city}?format=j1"

    # Map wttr.in weather codes to readable descriptions
    _WX_CODES = {
        "113": "Sunny ☀️",
        "116": "Partly Cloudy ⛅",
        "119": "Cloudy ☁️",
        "122": "Overcast ☁️",
        "143": "Mist 🌫️",
        "176": "Patchy Rain 🌦️",
        "179": "Patchy Snow 🌨️",
        "182": "Patchy Sleet 🌧️",
        "185": "Patchy Freezing Drizzle 🌧️",
        "200": "Thundery Outbreaks ⛈️",
        "227": "Blowing Snow 🌨️",
        "230": "Blizzard ❄️",
        "248": "Fog 🌫️",
        "260": "Freezing Fog 🌫️",
        "263": "Light Drizzle 🌦️",
        "266": "Drizzle 🌧️",
        "281": "Freezing Drizzle 🌧️",
        "284": "Heavy Freezing Drizzle 🌧️",
        "293": "Light Rain 🌧️",
        "296": "Light Rain 🌧️",
        "299": "Moderate Rain 🌧️",
        "302": "Moderate Rain 🌧️",
        "305": "Heavy Rain 🌧️",
        "308": "Heavy Rain 🌧️",
        "311": "Light Sleet 🌧️",
        "314": "Moderate Sleet 🌧️",
        "317": "Light Sleet 🌧️",
        "320": "Moderate Snow 🌨️",
        "323": "Light Snow 🌨️",
        "326": "Light Snow 🌨️",
        "329": "Moderate Snow 🌨️",
        "332": "Moderate Snow 🌨️",
        "335": "Heavy Snow ❄️",
        "338": "Heavy Snow ❄️",
        "350": "Ice Pellets 🌨️",
        "353": "Light Shower 🌦️",
        "356": "Heavy Shower 🌧️",
        "359": "Torrential Rain 🌧️",
        "362": "Light Sleet Shower 🌧️",
        "365": "Moderate Sleet Shower 🌧️",
        "368": "Light Snow Shower 🌨️",
        "371": "Heavy Snow Shower ❄️",
        "374": "Light Ice Pellet Shower 🌨️",
        "377": "Moderate Ice Pellet Shower 🌨️",
        "386": "Thundery Rain ⛈️",
        "389": "Heavy Thundery Rain ⛈️",
        "392": "Thundery Snow ⛈️",
        "395": "Heavy Thundery Snow ⛈️",
    }

    def get_weather(self, city: str) -> str:
        city = city.strip().strip('"\'')
        if not city:
            return "Please provide a city name."
        try:
            url = self.BASE_URL.format(city=urllib.parse.quote(city))
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "ARIA-VoiceAssistant/2.0"},
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode())

            current = data["current_condition"][0]
            area    = data["nearest_area"][0]

            # Location name
            area_name    = area["areaName"][0]["value"]
            country_name = area["country"][0]["value"]
            location_str = f"{area_name}, {country_name}"

            # Weather values
            temp_c    = current["temp_C"]
            temp_f    = current["temp_F"]
            feels_c   = current["FeelsLikeC"]
            feels_f   = current["FeelsLikeF"]
            humidity  = current["humidity"]
            wind_kmph = current["windspeedKmph"]
            wind_dir  = current["winddir16Point"]
            visibility= current["visibility"]
            uv_index  = current.get("uvIndex", "N/A")
            desc_code = current["weatherCode"]
            desc      = self._WX_CODES.get(desc_code, current["weatherDesc"][0]["value"])

            # Today's forecast (high/low)
            today     = data["weather"][0]
            max_c     = today["maxtempC"]
            min_c     = today["mintempC"]
            max_f     = today["maxtempF"]
            min_f     = today["mintempF"]
            sunrise   = today["astronomy"][0]["sunrise"]
            sunset    = today["astronomy"][0]["sunset"]

            return (
                f"Weather in {location_str}:\n"
                f"Condition  : {desc}\n"
                f"Temperature: {temp_c}°C / {temp_f}°F (feels like {feels_c}°C / {feels_f}°F)\n"
                f"Today      : High {max_c}°C / {max_f}°F — Low {min_c}°C / {min_f}°F\n"
                f"Humidity   : {humidity}%\n"
                f"Wind       : {wind_kmph} km/h {wind_dir}\n"
                f"Visibility : {visibility} km\n"
                f"UV Index   : {uv_index}\n"
                f"Sunrise    : {sunrise}  |  Sunset: {sunset}"
            )

        except urllib.error.URLError as exc:
            return f"Weather unavailable — network error: {exc.reason}"
        except (KeyError, json.JSONDecodeError) as exc:
            return f"Couldn't parse weather data for '{city}': {exc}"
        except Exception as exc:
            return f"Weather error: {exc}"


# ══════════════════════════════════════════════════════════
#  5. Time Tool — current date/time for any timezone/city
# ══════════════════════════════════════════════════════════
class TimeTool:
    """
    Returns the current date and time.
    - No argument / 'local'  → server local time
    - City name              → attempts timezone lookup via timeapi.io
    - Falls back gracefully if network is unavailable.
    """

    # Rough city → timezone mapping for common cities (offline fallback)
    _CITY_TZ: dict[str, str] = {
        "lucknow":      "Asia/Kolkata",
        "delhi":        "Asia/Kolkata",
        "new delhi":    "Asia/Kolkata",
        "mumbai":       "Asia/Kolkata",
        "bangalore":    "Asia/Kolkata",
        "kolkata":      "Asia/Kolkata",
        "chennai":      "Asia/Kolkata",
        "hyderabad":    "Asia/Kolkata",
        "pune":         "Asia/Kolkata",
        "ahmedabad":    "Asia/Kolkata",
        "jaipur":       "Asia/Kolkata",
        "new york":     "America/New_York",
        "london":       "Europe/London",
        "paris":        "Europe/Paris",
        "berlin":       "Europe/Berlin",
        "tokyo":        "Asia/Tokyo",
        "beijing":      "Asia/Shanghai",
        "shanghai":     "Asia/Shanghai",
        "dubai":        "Asia/Dubai",
        "sydney":       "Australia/Sydney",
        "los angeles":  "America/Los_Angeles",
        "chicago":      "America/Chicago",
        "toronto":      "America/Toronto",
        "singapore":    "Asia/Singapore",
        "moscow":       "Europe/Moscow",
        "istanbul":     "Europe/Istanbul",
        "cairo":        "Africa/Cairo",
        "nairobi":      "Africa/Nairobi",
        "karachi":      "Asia/Karachi",
        "dhaka":        "Asia/Dhaka",
        "kathmandu":    "Asia/Kathmandu",
        "colombo":      "Asia/Colombo",
    }

    def get_time(self, location: str = "") -> str:
        import datetime

        location = location.strip().strip('"\'').lower()

        # Try to resolve timezone
        tz_name = self._resolve_timezone(location)

        try:
            import zoneinfo
            tz = zoneinfo.ZoneInfo(tz_name) if tz_name else None
        except Exception:
            try:
                import pytz
                tz = pytz.timezone(tz_name) if tz_name else None
            except Exception:
                tz = None

        now = datetime.datetime.now(tz=tz)
        fmt_date = now.strftime("%A, %d %B %Y")
        fmt_time = now.strftime("%I:%M %p")
        fmt_tz   = tz_name if tz_name else "local server time"

        if location and location not in ("local", "server", ""):
            label = location.title()
        else:
            label = "Local"

        return (
            f"Current time for {label}:\n"
            f"Date : {fmt_date}\n"
            f"Time : {fmt_time}\n"
            f"Zone : {fmt_tz}"
        )

    def _resolve_timezone(self, location: str) -> str | None:
        if not location or location in ("local", "server"):
            return None
        # 1. Check offline map first
        if location in self._CITY_TZ:
            return self._CITY_TZ[location]
        # 2. Try timeapi.io for unknown cities
        try:
            city_encoded = urllib.parse.quote(location)
            url = f"https://timeapi.io/api/TimeZone/zone?timeZone={city_encoded}"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "ARIA-VoiceAssistant/2.0"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                return data.get("timeZone")
        except Exception:
            pass
        # 3. Fall back to local time
        return None


# ══════════════════════════════════════════════════════════
#  ToolRegistry — single access point
# ══════════════════════════════════════════════════════════
class ToolRegistry:
    def __init__(self) -> None:
        self.search  = WebSearchTool()
        self.docs    = DocumentQATool()
        self.code    = CodeExecTool()
        self.weather = WeatherTool()
        self.time    = TimeTool()

    def describe(self) -> str:
        """Returns capability description injected into system prompt."""
        return textwrap.dedent("""
            AVAILABLE TOOLS (use when needed, invoke by including the exact tag):

            1. WEB_SEARCH — To search the internet for current info:
               Output format: <tool:search>your query here</tool:search>

            2. DOC_QUERY — To answer questions from uploaded documents:
               Output format: <tool:doc>your question here</tool:doc>

            3. CODE_EXEC — To run Python code and get the result:
               Output format: <tool:code>
               ```python
               print("hello")
               ```
               </tool:code>

            4. WEATHER — To get live weather for any city:
               Output format: <tool:weather>city name</tool:weather>
               Example: <tool:weather>Lucknow</tool:weather>
               Use this whenever the user asks about weather, temperature,
               humidity, rain, forecast, or climate conditions for any location.

            5. TIME — To get the current date and time for any city or timezone:
               Output format: <tool:time>city name</tool:time>
               Example: <tool:time>Tokyo</tool:time>
               For local/server time: <tool:time>local</tool:time>
               Use this whenever the user asks what time or date it is,
               anywhere in the world.

            Rules:
            - Use web search for current events, facts, prices, news.
            - Use doc query when the user references an uploaded file.
            - Use code exec for math, data tasks, or when asked to run code.
            - ALWAYS use the weather tool for weather questions — never guess.
            - ALWAYS use the time tool for time/date questions — never guess.
            - After getting a tool result, incorporate it naturally into your reply.
            - Never fabricate tool results.
        """).strip()