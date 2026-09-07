"""The tenant record.

Field-by-field, this is the answer to "what does client two have to change?".
If onboarding a business ever requires editing anything outside this file's
data, that thing belongs in this file instead.

The knowledge-base fields (:class:`DayHours`, :class:`Service`, :class:`FAQ`,
:class:`EmergencyPolicy`) exist because the prompt says "your knowledge base has
the service area, hours, services, pricing, FAQs and emergency policy. Use it.
Never state a fact that isn't in it." -- and until these were added, there was
no such thing. The model was being told not to invent facts while being given
none, which is the setup for exactly the hallucination the `hallucination_kb`
eval grades against.

These are rendered inline into the system prompt rather than retrieved. A
handful of hours and price ranges fits comfortably in the context window, and
retrieval would add a lookup to the critical path to fetch text that could have
just been there.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def _spoken_time(hhmm: str) -> str:
    """'08:00' -> 'eight', '17:30' -> 'five-thirty'. Spoken, never digits."""
    words = {
        0: "twelve", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
        7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
    }
    tens = {
        15: "fifteen", 30: "thirty", 45: "forty-five",
    }
    hour, _, minute = hhmm.partition(":")
    h = int(hour) % 12
    m = int(minute or 0)
    spoken = words[h if h else 12]
    if m:
        spoken = f"{spoken}-{tens.get(m, str(m))}"
    return spoken


class Staff(BaseModel):
    """A named human the agent may mention or take messages for."""

    name: str
    role: str = ""


class BookingWindow(BaseModel):
    """An arrival window. Never an exact clock time -- trades don't work that way."""

    label: str = Field(description="Spoken form, e.g. 'eight to eleven'")
    start: str = Field(description="24h local, e.g. '08:00'")
    end: str = Field(description="24h local, e.g. '11:00'")


class DayHours(BaseModel):
    """When the office is open on one weekday. `open_at` None means closed."""

    day: str
    open_at: str | None = Field(default=None, description="24h local, e.g. '08:00'")
    close_at: str | None = Field(default=None, description="24h local, e.g. '17:00'")

    @field_validator("day")
    @classmethod
    def _known_day(cls, v: str) -> str:
        if v.lower() not in WEEKDAYS:
            raise ValueError(f"day must be one of {WEEKDAYS}, got {v!r}")
        return v.lower()

    @property
    def closed(self) -> bool:
        return self.open_at is None or self.close_at is None

    @property
    def spoken(self) -> str:
        if self.closed:
            return "closed"
        return f"{_spoken_time(self.open_at or '')} to {_spoken_time(self.close_at or '')}"


class Service(BaseModel):
    """Something the business does, and what it costs.

    Prices are ranges by construction -- there is no field for a firm number,
    because the prompt forbids quoting one and a data model that can express a
    forbidden state will eventually be asked to.
    """

    name: str
    price_low: int | None = Field(default=None, description="USD, spoken as words")
    price_high: int | None = Field(default=None, description="USD, spoken as words")
    note: str = Field(default="", description="One clause the agent may add")

    @property
    def price_spoken(self) -> str:
        if self.price_low is None or self.price_high is None:
            return "priced after the tech takes a look"
        return f"somewhere between {self.price_low} and {self.price_high} dollars"


class FAQ(BaseModel):
    """A question the agent is allowed to answer without checking with anyone."""

    question: str
    answer: str


class EmergencyPolicy(BaseModel):
    """What counts as an emergency at this business.

    The *handling* of an emergency is in the prompt and is identical everywhere.
    What differs per business is the trigger list, so only that lives here.
    """

    conditions: list[str] = Field(
        default_factory=list,
        description="Spoken conditions, e.g. 'no heat when it is below freezing'",
    )
    after_hours: bool = Field(
        default=True, description="Whether the on-call tech is dispatched outside hours"
    )


class Reseller(BaseModel):
    """Who takes prospect callbacks when this tenant runs in demo mode."""

    company: str
    product_name: str
    contact_full_name: str
    contact_spoken_name: str = Field(description="How the agent refers to them after first use")
    callback_days: str = Field(description="Spoken, e.g. 'Monday to Friday'")
    callback_hours: str = Field(description="Spoken, e.g. 'eight in the morning to five-thirty'")
    callback_timezone: str = Field(description="Spoken, e.g. 'Eastern'")


class Tenant(BaseModel):
    """One business the agent can answer as.

    `demo_mode` is the switch that decides whether this is a working receptionist
    or the self-selling demo line. It gates the prospect, taste, and close blocks
    of the prompt as a unit -- those three only make sense together, and a real
    client must never get any of them.
    """

    tenant_id: str
    phone_number: str = Field(description="E.164, the number callers dial")

    business_name: str
    trade: str = Field(description="e.g. 'HVAC', 'roofing', 'plumbing'")
    city: str
    state: str
    timezone: str = "America/New_York"

    agent_name: str = Field(description="The persona's own name, e.g. 'Casey'")
    agent_persona: str = Field(description="One or two lines of character, spoken about in 2nd person")

    service_area: list[str] = Field(default_factory=list, description="Towns served")
    staff: list[Staff] = Field(default_factory=list)
    booking_windows: list[BookingWindow] = Field(default_factory=list)

    hours: list[DayHours] = Field(default_factory=list)
    services: list[Service] = Field(default_factory=list)
    faqs: list[FAQ] = Field(default_factory=list)
    emergency_policy: EmergencyPolicy = Field(default_factory=EmergencyPolicy)

    capacity_per_window: int = Field(
        default=2, description="Appointments bookable in one window. Usually the tech count."
    )

    asr_keywords: list[str] = Field(
        default_factory=list,
        description="Boosted terms: town names, staff names, the agent name. Cheap accuracy.",
    )

    demo_mode: bool = False
    reseller: Reseller | None = None

    voice_id: str = Field(description="TTS vendor voice id")
    greeting: str = Field(description="Verbatim first message, including any AI/recording notice")

    recording_disclosure_required: bool = True
    """Per-tenant, never global: consent law is per-jurisdiction and this is multi-state."""

    sms_consent_required: bool = True
    """Enforced server-side at send time, never merely asked for in the prompt."""

    @field_validator("phone_number")
    @classmethod
    def _e164(cls, v: str) -> str:
        if not v.startswith("+") or not v[1:].isdigit():
            raise ValueError(f"phone_number must be E.164, got {v!r}")
        return v

    def model_post_init(self, _ctx: object) -> None:
        if self.demo_mode and self.reseller is None:
            raise ValueError(
                f"tenant {self.tenant_id!r}: demo_mode needs a reseller -- "
                "the close block has nobody to hand the lead to"
            )
        seen = {h.day for h in self.hours}
        if self.hours and len(seen) != len(self.hours):
            raise ValueError(f"tenant {self.tenant_id!r}: a weekday is listed twice in hours")

    @property
    def service_area_spoken(self) -> str:
        """Oxford-comma list for use inside prompt prose."""
        towns = self.service_area
        if len(towns) <= 1:
            return "".join(towns)
        if len(towns) == 2:
            return f"{towns[0]} and {towns[1]}"
        return f"{', '.join(towns[:-1])}, and {towns[-1]}"

    @property
    def windows_spoken(self) -> str:
        """e.g. 'eight to eleven, eleven to two, or two to five'."""
        labels = [w.label for w in self.booking_windows]
        if len(labels) <= 1:
            return "".join(labels)
        return f"{', '.join(labels[:-1])}, or {labels[-1]}"

    def hours_for(self, day: str) -> DayHours | None:
        """Hours for a weekday name, or None if the tenant hasn't declared any."""
        day = day.lower()
        return next((h for h in self.hours if h.day == day), None)

    def is_open_on(self, day: str) -> bool:
        """Whether the office is open on `day`. Unknown days count as open.

        Unknown means the tenant hasn't filled in hours yet; refusing every
        booking would be a worse failure than offering one that gets confirmed
        by the office in the morning.
        """
        h = self.hours_for(day)
        return True if h is None else not h.closed

    @property
    def hours_spoken(self) -> str:
        """'Monday to Friday, eight to five. Saturday, nine to one. Closed Sunday.'

        Consecutive days with identical hours are collapsed, because reading
        seven lines of hours down a phone line is not what a person does.
        """
        if not self.hours:
            return ""
        by_day = {h.day: h for h in self.hours}
        ordered = [by_day[d] for d in WEEKDAYS if d in by_day]

        groups: list[tuple[list[DayHours], str]] = []
        for h in ordered:
            spoken = h.spoken
            if groups and groups[-1][1] == spoken:
                groups[-1][0].append(h)
            else:
                groups.append(([h], spoken))

        parts: list[str] = []
        for days, spoken in groups:
            first, last = days[0].day.capitalize(), days[-1].day.capitalize()
            span = first if len(days) == 1 else f"{first} to {last}"
            parts.append(f"Closed {span}" if spoken == "closed" else f"{span}, {spoken}")
        return ". ".join(parts) + "."

    @property
    def services_spoken(self) -> list[str]:
        """One line per service, price ranges already in spoken form."""
        out = []
        for s in self.services:
            line = f"{s.name}: {s.price_spoken}"
            if s.note:
                line += f" ({s.note})"
            out.append(line)
        return out
