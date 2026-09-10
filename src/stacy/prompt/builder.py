"""Render the receptionist prompt for a given tenant.

Two rules keep this honest:

1.  Jinja runs in StrictUndefined. A template slot with no tenant value is a
    crash at boot, not a sentence with a hole in it at 2am on a live call.
2.  Delimiters are ``${var}`` and ``{% block %}``, never ``{{ }}``. The runtime
    placeholders ``{{system__time}}`` and ``{{system__caller_id}}`` survive
    rendering untouched and are filled per-turn by the agent.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from stacy.tenants.models import Tenant

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_DEFAULT_TEMPLATE = "receptionist.md.j2"

RUNTIME_PLACEHOLDERS = ("{{system__time}}", "{{system__caller_id}}")
"""Filled per-call, not at render time. Asserted present by :meth:`PromptBuilder.render`.

Both live in a `# This call` section at the very *end* of the template, and that
placement is load-bearing rather than stylistic. Prompt caching keys on a prefix:
everything before the first byte that varies is cacheable, everything after it is
not. These two lines used to sit in the Environment section about 200 tokens in,
which left roughly 5,200 tokens of stable rules re-prefilled on every turn of
every call. The timestamp also varies by the minute, so the small cacheable
prefix missed across calls as well.

Anything added to this template that changes per call belongs below them.
"""


def _spoken_list(items: list[str]) -> str:
    """'Dave, Lisa and Sam' -> 'Dave or Lisa or Sam' reads badly; use 'Dave or Lisa'."""
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} or {items[1]}"
    return f"{', '.join(items[:-1])}, or {items[-1]}"


class PromptBuilder:
    """Renders tenant records into system prompts."""

    def __init__(self, template_dir: str | Path = _TEMPLATE_DIR) -> None:
        self._env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            variable_start_string="${",
            variable_end_string="}",
            block_start_string="{%",
            block_end_string="%}",
            comment_start_string="{#",
            comment_end_string="#}",
            undefined=StrictUndefined,
            trim_blocks=False,
            lstrip_blocks=False,
            keep_trailing_newline=True,
        )

    def context(self, tenant: Tenant) -> dict[str, Any]:
        """Flatten a tenant into template variables.

        Flattening rather than passing the object keeps the template readable
        and means a missing field fails at render with the slot's name in the
        traceback.
        """
        staff_names = [s.name for s in tenant.staff]
        # Prefer an office contact for escalation; a tech in a crawlspace cannot
        # take an angry call. Falls back to the first named person, then a generic.
        office = next((s.name for s in tenant.staff if s.role == "office"), None)
        escalation = office or (staff_names[0] if staff_names else "the office")

        ctx: dict[str, Any] = {
            "agent_name": tenant.agent_name,
            "agent_persona": tenant.agent_persona,
            "business_name": tenant.business_name,
            "trade": tenant.trade,
            "trade_lowercase": tenant.trade.lower(),
            "city": tenant.city,
            "state": tenant.state,
            "service_area_spoken": tenant.service_area_spoken,
            "windows_spoken": tenant.windows_spoken,
            "staff_spoken": _spoken_list(staff_names),
            "escalation_staff": escalation,
            "demo_mode": tenant.demo_mode,
            # The knowledge base. The prompt has always told the model "never
            # state a fact that isn't in it" -- these are the first version of
            # this repo where there is an "it" to be in.
            "hours_spoken": tenant.hours_spoken,
            "services_lines": tenant.services_spoken,
            "faqs": [(f.question, f.answer) for f in tenant.faqs],
            "emergency_conditions": tenant.emergency_policy.conditions,
            "emergency_after_hours": tenant.emergency_policy.after_hours,
        }

        if tenant.reseller is not None:
            r = tenant.reseller
            ctx.update(
                {
                    "reseller_company": r.company,
                    # "QuickFlip Digital" -> "QuickFlip" for mid-sentence use.
                    "reseller_short_company": r.company.split()[0],
                    "reseller_product_name": r.product_name,
                    "reseller_full_name": r.contact_full_name,
                    "reseller_first_name": r.contact_full_name.split()[0],
                    "reseller_spoken_name": r.contact_spoken_name,
                    "reseller_callback_days": r.callback_days,
                    "reseller_callback_hours": r.callback_hours,
                    "reseller_callback_timezone": r.callback_timezone,
                }
            )
        return ctx

    def render(self, tenant: Tenant, template: str = _DEFAULT_TEMPLATE) -> str:
        """Render `tenant`'s system prompt.

        Raises if a runtime placeholder went missing -- losing
        ``{{system__caller_id}}`` would silently break the close, which is the
        one place the agent makes a promise it has to keep.
        """
        text = self._env.get_template(template).render(**self.context(tenant))
        for placeholder in RUNTIME_PLACEHOLDERS:
            if placeholder not in text:
                raise ValueError(
                    f"rendered prompt for {tenant.tenant_id!r} lost {placeholder} -- "
                    "the template was edited in a way that breaks per-turn substitution"
                )
        # Collapse the blank-line runs that conditional blocks leave behind.
        return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"

    def fill_runtime(self, prompt: str, *, now: str, caller_id: str) -> str:
        """Substitute the per-turn placeholders just before the call starts.

        An unknown caller id becomes an empty string, not the literal "None":
        the prompt explicitly branches on "if nothing appears after 'is'".
        """
        return prompt.replace("{{system__time}}", now).replace(
            "{{system__caller_id}}", caller_id or ""
        )


def render_system_prompt(tenant: Tenant) -> str:
    """One-shot convenience wrapper."""
    return PromptBuilder().render(tenant)
