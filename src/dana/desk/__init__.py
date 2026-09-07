"""What a front desk actually owns: the calendar and the message pad.

Both are real stores with real refusals -- a full window declines, a booking
without a street address declines. The prompt describes those rules; these
modules are what makes them true when the model has a bad turn.
"""

from dana.desk.calendar import (
    Appointment,
    BookingError,
    CalendarStore,
    SlotUnavailable,
    UnknownWindow,
    resolve_day,
)
from dana.desk.messages import Message, MessageStore, Priority, classify_priority

__all__ = [
    "Appointment",
    "BookingError",
    "CalendarStore",
    "Message",
    "MessageStore",
    "Priority",
    "SlotUnavailable",
    "UnknownWindow",
    "classify_priority",
    "resolve_day",
]
