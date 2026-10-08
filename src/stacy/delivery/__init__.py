"""Making the agent's promises true after the call ends.

Two promises get made on nearly every call: someone will get your details, and
you'll get a text. Neither is kept by saying it.
"""

from stacy.delivery.archive import CallArchive
from stacy.delivery.sms import ConsentMissing, LoggingProvider, SmsProvider, SmsSender
from stacy.delivery.webhook import (
    RETRY_DELAYS_S,
    SIGNATURE_HEADER,
    WebhookSender,
    call_payload,
    sign,
    verify,
)

__all__ = [
    "RETRY_DELAYS_S",
    "SIGNATURE_HEADER",
    "CallArchive",
    "ConsentMissing",
    "LoggingProvider",
    "SmsProvider",
    "SmsSender",
    "WebhookSender",
    "call_payload",
    "sign",
    "verify",
]
