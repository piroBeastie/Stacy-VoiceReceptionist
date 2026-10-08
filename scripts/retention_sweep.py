"""Delete the plaintext that is older than the retention policy allows.

    python scripts/retention_sweep.py              # say what would go
    python scripts/retention_sweep.py --apply      # actually delete it

Prints what it would do and exits without touching anything unless `--apply` is
passed, because the records belong to a business and its customers. Schedule it
(cron, Task Scheduler, a sidecar) once the windows in `stacy.retention` match
the client's own policy. See that module for the defaults and why they are those.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stacy.retention import RetentionPolicy, sweep  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.getenv("STACY_DATA_DIR", "data"))
    ap.add_argument("--apply", action="store_true", help="Delete, rather than report")
    ap.add_argument("--quiet", action="store_true", help="Only print the summary line")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s")
    policy, chosen = RetentionPolicy.from_env()
    print(
        f"policy: calls {policy.calls_days}d, traces {policy.traces_days}d, "
        f"messages {policy.messages_days}d, appointments "
        f"{policy.appointments_days or 'kept'}"
        + (f" (set by {', '.join(chosen)})" if chosen else " (defaults)")
    )

    result = sweep(args.data_dir, policy, apply=args.apply)
    if not args.quiet:
        for path in result.files_removed:
            print(f"  file   {path}")
        for path, count in result.lines_removed.items():
            print(f"  {count:>5} record(s) from {path}")
    print(result.spoken())
    if not args.apply and not result.nothing_to_do:
        print("nothing was deleted; pass --apply to do it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
