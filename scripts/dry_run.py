"""Walk a scripted call through the real stores, with no SIP trunk and no model.

    python scripts/dry_run.py [--tenant millbrook]

Every line printed under "agent" is a real return value from the tool the agent
would have called, against the real calendar. This is not a simulation of the
model -- it is the half of the call the model doesn't control, exercised on its
own, which is the half that turns a good-sounding answer into a visit somebody
actually shows up for.

Useful for checking tool wording out loud before a phone is involved: if a
refusal reads badly here, it reads worse at 8pm to somebody whose heat is out.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stacy.agent import CallRuntime, DeskOperations  # noqa: E402
from stacy.delivery import LoggingProvider, SmsSender, sign, verify  # noqa: E402
from stacy.delivery.sms import ConsentMissing  # noqa: E402
from stacy.desk import CalendarStore, MessageStore  # noqa: E402
from stacy.tenants import TenantStore  # noqa: E402
import json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tenant", default="millbrook")
    ap.add_argument("--tenant-dir", default="src/stacy/tenants/data")
    ap.add_argument(
        "--keep",
        action="store_true",
        help="Write into STACY_DATA_DIR instead of a throwaway directory",
    )
    args = ap.parse_args()

    # The prompt and every tool return value are full of em dashes; a Windows
    # console defaults to cp1252 and renders them as replacement characters,
    # which makes the wording look broken when it isn't.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Every refusal below is deliberate and is shown in the transcript. Their
    # log warnings would otherwise interleave with it on stderr, out of order.
    logging.disable(logging.WARNING)

    tenant = TenantStore(args.tenant_dir).get(args.tenant)
    root = Path("data") if args.keep else Path(tempfile.mkdtemp(prefix="stacy-dryrun-"))
    calendar = CalendarStore(root / "calendar")
    messages = MessageStore(root / "messages")

    def new_call(caller_id: str) -> tuple[CallRuntime, DeskOperations]:
        """One runtime per caller. Sharing one across callers is what hid the
        bug where a second booking overwrote the first."""
        rt = CallRuntime.build(
            tenant=tenant,
            call_id=f"dry-run-{caller_id[-4:]}",
            caller_id=caller_id,
            calendar=calendar,
            messages=messages,
            now=datetime.now(ZoneInfo(tenant.timezone)),
        )
        return rt, DeskOperations(rt)

    runtime, ops = new_call("+12035550147")

    def turn(caller: str, said: str) -> None:
        print(f"  caller  {caller}")
        print(f"  agent   {said}\n")

    print(f"\n{tenant.business_name} — dry run against {root}\n")
    print(f"  agent   {tenant.greeting}\n")

    turn("(caller id arrives)", ops.lookup_customer())

    # The thing callers actually do: three fields in one breath.
    turn(
        "Yeah hi, this is Dan over at fourteen Elm in Wallingford and my furnace "
        "is making a noise.",
        ops.record_details(
            customer_name="Dan",
            street_address="14 Elm Street",
            town="Wallingford",
            problem="furnace making a noise",
            system_type="furnace",
        ),
    )
    turn(
        "Two-oh-three, five-five-five, oh-one-four-seven. And yeah, texts are fine.",
        ops.record_details(
            phone="two zero three five five five oh one four seven", sms_consent=True
        ),
    )
    turn("It's gas. Can someone come tomorrow?", ops.record_details(fuel="gas"))
    turn("What have you got?", ops.check_availability("tomorrow"))
    turn(
        "Eleven to two works.",
        ops.record_details(day="tomorrow", window="eleven to two"),
    )
    turn("Yep, that's right.", ops.book_appointment())

    print("  -- delivery for that call --\n")
    sender = SmsSender(provider=LoggingProvider())
    for appt in runtime.appointments:
        try:
            sender.send_confirmation(tenant, appt)
            print(f"  sms     {sender.confirmation_body(tenant, appt)}")
        except ConsentMissing as exc:
            print(f"  sms     refused: {exc}")
    print()

    print("  -- next caller: the window is now full, and the refusals --\n")
    filler_runtime, filler_ops = new_call("+12035550200")
    for i in range(tenant.capacity_per_window - 1):
        filler_ops.book_appointment(
            customer_name=f"Filler {i}",
            phone=f"+1203555020{i}",
            street_address=f"{i} Main Street",
            town="Wallingford",
            problem="tune up",
            day="tomorrow",
            window="eleven to two",
        )

    runtime, ops = new_call("+12035550188")
    turn(
        "Eleven to two, please.",
        ops.book_appointment(
            customer_name="Ruth",
            phone="+12035550188",
            street_address="9 Oak Lane",
            town="Meriden",
            problem="no cooling",
            day="tomorrow",
            window="eleven to two",
        ),
    )
    turn(
        "Book me for seven to nine, then.",
        ops.book_appointment(
            customer_name="Ruth",
            phone="+12035550188",
            street_address="9 Oak Lane",
            town="Meriden",
            problem="no cooling",
            day="tomorrow",
            window="seven to nine",
        ),
    )
    turn(
        "Fine, two to five. I'll give you the address later.",
        ops.book_appointment(
            customer_name="Ruth",
            phone="+12035550188",
            street_address="",
            town="Meriden",
            problem="no cooling",
            day="tomorrow",
            window="two to five",
        ),
    )
    turn("Can I leave a message for Lisa instead?", ops.take_message(
        caller_name="Ruth",
        phone="+12035550188",
        summary="wants a quote on a new furnace",
        for_staff="Lisa",
    ))
    print(f"  (filler call outcome: {filler_runtime.outcome})\n")
    print("  -- delivery --\n")

    payload = runtime.post_call_payload()
    body = json.dumps(payload, separators=(",", ":")).encode()
    header = sign(body, "whsec_dry_run")
    print(f"  webhook outcome={payload['outcome']} bytes={len(body)}")
    print(f"  webhook signature verifies: {verify(body, header, 'whsec_dry_run')}")
    if not args.keep:
        print(f"\n(throwaway data in {root} — pass --keep to write into data/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
