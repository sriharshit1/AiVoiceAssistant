"""
memory.py — Short-term session memory for AURA.

Tracks:
  - Full conversation turns (role + content)
  - Named facts extracted per turn  (name, preferences, topics mentioned)
  - Turn count and session start time
  - A rolling summary injected into the system prompt every N turns

Nothing is written to disk — all state lives in-process.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Literal


Role = Literal["user", "assistant"]


@dataclass
class Turn:
    role: Role
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class MemoryFact:
    """A single extracted fact about the user."""
    key: str        # e.g. "name", "likes", "dislikes", "location"
    value: str
    turn_index: int


class ShortTermMemory:
    """
    Manages conversation history + lightweight fact extraction.

    Usage
    -----
    mem = ShortTermMemory(max_turns=20, summary_every=8)
    mem.add("user", "My name is Harshit")
    mem.add("assistant", "Nice to meet you, Harshit!")
    messages = mem.build_messages(system_prompt)   # → list[dict] for Groq
    """

    # Simple regex patterns for on-the-fly fact extraction
    _FACT_PATTERNS: list[tuple[str, re.Pattern]] = [
        ("name",     re.compile(r"\bmy name is ([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)", re.I)),
        ("name",     re.compile(r"\bcall me ([A-Z][a-z]+)", re.I)),
        ("location", re.compile(r"\bI(?:'m| am) (?:from|in|at) ([A-Za-z ,]+?)[\.,!?]", re.I)),
        ("likes",    re.compile(r"\bI (?:love|like|enjoy|prefer) ([^\.!?]+)", re.I)),
        ("dislikes", re.compile(r"\bI (?:hate|dislike|don't like) ([^\.!?]+)", re.I)),
        ("language", re.compile(r"\bI(?:'m| am) (?:using|coding in|working with) ([A-Za-z+#]+)", re.I)),
    ]

    def __init__(self, max_turns: int = 20, summary_every: int = 8) -> None:
        self.max_turns = max_turns
        self.summary_every = summary_every
        self.session_start = time.time()

        self._turns: list[Turn] = []
        self._facts: dict[str, str] = {}   # key → latest value
        self._rolling_summary: str = ""

    # ── public API ────────────────────────────────────────

    def add(self, role: Role, content: str) -> None:
        """Record a turn and extract facts from user speech."""
        self._turns.append(Turn(role=role, content=content))
        if role == "user":
            self._extract_facts(content, turn_index=len(self._turns) - 1)
        # Trim to max_turns (keep most recent), never drop the first system context
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns:]
        # Periodically build a rolling summary to compress old context
        if len(self._turns) % self.summary_every == 0:
            self._update_summary()

    def build_messages(self, base_system_prompt: str) -> list[dict]:
        """
        Return a messages list ready for the Groq API.
        Injects known facts + rolling summary into the system prompt.
        """
        enriched_system = base_system_prompt
        memory_block = self._build_memory_block()
        if memory_block:
            enriched_system += f"\n\n---\nSESSION MEMORY (use naturally, don't recite):\n{memory_block}\n---"

        messages: list[dict] = [{"role": "system", "content": enriched_system}]
        messages += [{"role": t.role, "content": t.content} for t in self._turns]
        return messages

    def get_fact(self, key: str) -> str | None:
        return self._facts.get(key)

    def all_facts(self) -> dict[str, str]:
        return dict(self._facts)

    def turn_count(self) -> int:
        return len(self._turns)

    def session_duration(self) -> str:
        elapsed = int(time.time() - self.session_start)
        m, s = divmod(elapsed, 60)
        return f"{m}m {s}s" if m else f"{s}s"

    def reset(self) -> None:
        self._turns.clear()
        self._facts.clear()
        self._rolling_summary = ""

    # ── private helpers ───────────────────────────────────

    def _extract_facts(self, text: str, turn_index: int) -> None:
        for key, pattern in self._FACT_PATTERNS:
            match = pattern.search(text)
            if match:
                value = match.group(1).strip().rstrip(".,!?")
                # Don't overwrite a name with a weaker match
                if key == "name" and "name" in self._facts:
                    continue
                self._facts[key] = value

    def _build_memory_block(self) -> str:
        lines: list[str] = []
        if self._rolling_summary:
            lines.append(f"Earlier in this session: {self._rolling_summary}")
        if self._facts:
            fact_str = ", ".join(f"{k}={v}" for k, v in self._facts.items())
            lines.append(f"Known about user: {fact_str}")
        return "\n".join(lines)

    def _update_summary(self) -> None:
        """
        Build a very short plain-text summary of the oldest half of turns
        so we can compress them without losing context.
        This is heuristic-only (no LLM call) to stay fast.
        """
        half = len(self._turns) // 2
        old_turns = self._turns[:half]
        topics: list[str] = []
        for t in old_turns:
            # Pull the first meaningful noun phrase (first 8 words of user turns)
            if t.role == "user":
                snippet = " ".join(t.content.split()[:8])
                topics.append(snippet)
        if topics:
            self._rolling_summary = "; ".join(topics[-4:])   # keep last 4 snippets