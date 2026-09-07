"""The tenant record.

Field-by-field, this is the answer to "what does client two have to change?".
If onboarding a business ever requires editing anything outside this file's
data, that thing belongs in this file instead.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Staff(BaseModel):
    """A named human the agent may mention or take messages for."""

    name: str
    role: str = ""


class BookingWindow(BaseModel):
    """An arrival window. Never an exact clock time -- trades don't work that way."""

    label: str = Field(description="Spoken form, e.g. 'eight to eleven'")
    start: str = Field(description="24h local, e.g. '08:00'")
    end: str = Field(description="24h local, e.g. '11:00'")


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
