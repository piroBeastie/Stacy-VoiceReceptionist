"""The message pad.

Two of the three outcomes the prompt allows are "answer the question" and "book
an appointment". This is the third, and it is the fallback for the other two:
the error-handling section says that when a booking can't be completed, take a
message instead. That only works if there is somewhere for the message to go.

Priority is set by the caller's words, not by the model's mood -- see
:func:`classify_priority`. A model asked "how urgent is this?" will drift; a
keyword list will not, and when it is wrong it is wrong the same way every time,
which is the kind of wrong you can fix.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

Priority = Literal["emergency", "urgent", "normal"]

_EMERGENCY = re.compile(
    r"\b(gas|carbon monoxide|co alarm|smoke|fire|flood(ing|ed)?|burst|no heat|"
    r"emergency|911|leak(ing)?)\b",
    re.I,
)
_URGENT = re.compile(
    r"\b(today|tonight|right away|asap|as soon as possible|urgent|reschedul\w*|"
    r"cancel|tomorrow morning|freezing|elderly|newborn|baby)\b",
    re.I,
)


def classify_priority(text: str) -> Priority:
    """Bucket a message by what the caller actually said.

    Mirrors the platform's three levels. The ordering is deliberate: anything
    matching the emergency list is an emergency even if it also reads as
    routine, because the cost of under-calling one is not symmetric with the
    cost of over-calling one.
    """
    if _EMERGENCY.search(text):
        return "emergency"
    if _URGENT.search(text):
        return "urgent"
    return "normal"


@dataclass(frozen=True)
class Message:
    """One thing a caller wants passed to a human."""

    message_id: str
    tenant_id: str
    caller_name: str
    phone: str
    summary: str
    priority: Priority
    for_staff: str = ""
    """Named person the message is for, if the caller asked for one."""

    created_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat())


class MessageStore:
    """Append-only message pad, one JSONL per tenant."""

    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, tenant_id: str) -> Path:
        return self._dir / f"messages-{tenant_id}.jsonl"

    def take(
        self,
        *,
        tenant_id: str,
        caller_name: str,
        phone: str,
        summary: str,
        for_staff: str = "",
        priority: Priority | None = None,
    ) -> Message:
        """Record a message. Priority is derived unless the caller overrides it."""
        msg = Message(
            message_id=uuid.uuid4().hex[:12],
            tenant_id=tenant_id,
            caller_name=caller_name,
            phone=phone,
            summary=summary,
            priority=priority or classify_priority(summary),
            for_staff=for_staff,
        )
        with self._lock, self._path(tenant_id).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(msg), separators=(",", ":")) + "\n")
            fh.flush()
        return msg

    def all_for(self, tenant_id: str) -> list[Message]:
        """Every message for a tenant, newest first."""
        path = self._path(tenant_id)
        if not path.exists():
            return []
        out: list[Message] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(Message(**json.loads(line)))
                except (json.JSONDecodeError, TypeError):
                    continue
        # Reverse first so append order breaks ties: the clock on Windows has
        # ~15ms resolution and two messages taken in one turn share a timestamp.
        # For an append-only log the file order is the authoritative ordering.
        return sorted(  # noqa: C414 - reversed() is what breaks ties by append order
            reversed(out), key=lambda m: m.created_at, reverse=True
        )
