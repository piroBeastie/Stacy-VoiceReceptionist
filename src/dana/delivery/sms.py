"""Outbound SMS, and the consent gate that is the actual control.

The prompt makes the agent read a consent script and wait for a yes. That script
is a good thing to say and it is not a control: a prompt is a suggestion to a
model, and the model will occasionally skip it, mishear a "no", or be talked
past by a caller in a hurry. TCPA exposure does not care which of those
happened.

So the refusal lives here, at the send, where it is a branch rather than an
instruction. :class:`ConsentMissing` is raised on any send for an appointment
whose `sms_consent` is False, and there is no keyword argument to override it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from dana.desk.calendar import Appointment
from dana.tenants.models import Tenant

log = logging.getLogger("dana.delivery.sms")


class ConsentMissing(Exception):
    """A send was attempted for a booking that never captured consent."""


class SmsProvider(Protocol):
    """Anything that can put a text message on the wire."""

    def send(self, *, to: str, body: str) -> str: ...


class LoggingProvider:
    """Writes the message to the log instead of sending it.

    The default on purpose. An unconfigured provider that silently no-ops is
    indistinguishable from a working one until a customer says they never got
    the text; this one is visible in the worker log and obviously not a carrier.
    """

    def send(self, *, to: str, body: str) -> str:
        log.info("SMS (not sent -- no provider configured) to=%s body=%r", to, body)
        return "logged"


@dataclass
class SmsSender:
    """Sends confirmations, subject to consent."""

    provider: SmsProvider

    def confirmation_body(self, tenant: Tenant, appt: Appointment) -> str:
        """The confirmation text. Includes the opt-out because the script promised it."""
        when = (
            "today"
            if appt.kind == "emergency"
            else f"{appt.day}, {appt.window_label}"
        )
        return (
            f"{tenant.business_name}: you're confirmed for {when}. "
            f"{appt.street_address}, {appt.town}. Reply STOP to opt out."
        )

    def send_confirmation(self, tenant: Tenant, appt: Appointment) -> str:
        """Text the booking confirmation, or refuse.

        Raises:
            ConsentMissing: if consent was not captured on this booking and the
                tenant requires it. There is no bypass parameter, deliberately.
        """
        if tenant.sms_consent_required and not appt.sms_consent:
            raise ConsentMissing(
                f"appointment {appt.appointment_id} has no SMS consent; refusing to send"
            )
        return self.provider.send(to=appt.phone, body=self.confirmation_body(tenant, appt))
