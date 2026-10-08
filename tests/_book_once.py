"""Book one window from a separate process, and say what the calendar allowed.

Run by `TestTwoWorkersOneCalendar` rather than by pytest: the thing under test is
what a *second worker process* makes of the same calendar, which cannot be shown
from inside one interpreter. Prints exactly one word — `booked`, or the name of
the exception the store raised — so the caller can assert on it.

    python tests/_book_once.py <src-dir> <tenant-dir> <calendar-dir> <iso-day> <lock-wait-s>
"""

from __future__ import annotations

import sys
from datetime import date


def main(argv: list[str]) -> int:
    src, tenants, calendar_dir, iso_day, lock_wait = argv[1:6]
    sys.path.insert(0, src)

    from stacy.desk import calendar as calendar_module
    from stacy.tenants import TenantStore

    # Shortened by the caller when it wants the refusal rather than the wait.
    calendar_module.LOCK_WAIT_S = float(lock_wait)

    tenant = TenantStore(tenants).get("millbrook").model_copy(update={"capacity_per_window": 1})
    store = calendar_module.CalendarStore(calendar_dir)
    try:
        store.book(
            tenant,
            day=date.fromisoformat(iso_day),
            window_label="eight to eleven",
            customer_name="Race",
            phone="+12035550147",
            street_address="1 Elm Street",
            town="Wallingford",
            problem="no heat",
        )
    except Exception as exc:  # noqa: BLE001 - the exception's name is the result
        print(type(exc).__name__)
    else:
        print("booked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
