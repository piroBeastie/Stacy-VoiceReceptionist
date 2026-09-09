"""Generate the LiveKit SIP config from the tenant files, and print the commands.

    python scripts/sip_config.py                    # write deploy/livekit/*.json
    python scripts/sip_config.py --trunk-id ST_xxx  # fill the dispatch rule in

The number a caller dials is the routing key: :func:`dana.agent.entrypoint._route`
looks up the tenant by it and refuses to guess. So the number in the LiveKit
inbound trunk and the number in the tenant file have to be the same string, and
the way to guarantee that is to never type it twice. This reads the tenants and
emits the trunk config, so the two cannot drift.

They drift silently, which is the problem. A mismatch does not fail at deploy:
the call connects, LiveKit accepts it, a room is created, and the agent throws
KeyError while a customer listens to nothing.

THE JSON SHAPES BELOW ARE VERSION-SENSITIVE, like the SDK surfaces in
`agent/entrypoint.py`. They match the `lk` CLI's documented schema; if `lk`
rejects a field, that is a CLI version difference, not a bad number. Check with
`lk sip inbound create --help` before assuming the config is wrong.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dana.tenants import TenantStore  # noqa: E402

OUT = Path("deploy/livekit")


def inbound_trunk(numbers: list[str], name: str) -> dict:
    """One trunk carrying every tenant number.

    One trunk for all of them, not one each: the dispatch rule below routes on
    nothing, because the agent does its own routing from the dialed number. A
    trunk per tenant would be a second place to keep numbers in sync, which is
    the exact failure this script exists to prevent.

    `allowed_addresses` is left out deliberately. Restricting to Twilio's
    signaling ranges is real hardening, but those ranges change and pasting a
    stale list here would reject live calls. Fill it from Twilio's current
    published list when you harden, not from memory.
    """
    return {
        "trunk": {
            "name": name,
            "numbers": numbers,
        }
    }


def dispatch_rule(trunk_id: str, agent_name: str, name: str) -> dict:
    """A room per call, and -- if configured -- an explicit agent dispatch.

    `dispatchRuleIndividual` gives every call its own room. The alternative puts
    callers in a shared room, which for a receptionist would mean two customers
    hearing each other.

    With DANA_AGENT_NAME set the worker registers under that name and is
    dispatched only where it is named. Without it, the worker joins every room
    in the project automatically. Automatic is fewer moving parts; named is what
    you want the moment a second kind of agent shares the project.
    """
    rule: dict = {
        "name": name,
        "trunk_ids": [trunk_id] if trunk_id else [],
        "rule": {"dispatchRuleIndividual": {"roomPrefix": "call"}},
    }
    if agent_name:
        rule["room_config"] = {"agents": [{"agent_name": agent_name}]}
    return rule


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tenant-dir", default=os.getenv("DANA_TENANT_DIR", "src/dana/tenants/data"))
    ap.add_argument("--trunk-id", default="", help="ST_... from `lk sip inbound create`")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        from dotenv import load_dotenv  # noqa: PLC0415

        load_dotenv()
    except ImportError:
        pass

    tenants = TenantStore(args.tenant_dir).all()
    if not tenants:
        print(f"no tenants in {args.tenant_dir}", file=sys.stderr)
        return 1

    numbers = [t.phone_number for t in tenants]
    dupes = {n for n in numbers if numbers.count(n) > 1}
    if dupes:
        print(f"two tenants claim the same number: {sorted(dupes)}", file=sys.stderr)
        return 1

    agent_name = os.getenv("DANA_AGENT_NAME", "").strip()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    trunk_path = out / "inbound-trunk.json"
    rule_path = out / "dispatch-rule.json"
    trunk_path.write_text(
        json.dumps(inbound_trunk(numbers, "dana-inbound"), indent=2) + "\n", encoding="utf-8"
    )
    rule_path.write_text(
        json.dumps(dispatch_rule(args.trunk_id, agent_name, "dana-dispatch"), indent=2) + "\n",
        encoding="utf-8",
    )

    for tenant in tenants:
        print(f"  {tenant.phone_number}  ->  {tenant.tenant_id}  ({tenant.business_name})")
    print()
    print(f"wrote {trunk_path}")
    print(f"wrote {rule_path}")
    print()

    if agent_name:
        print(f"Dispatch is explicit: the worker must run with DANA_AGENT_NAME={agent_name!r},")
        print("and only rooms naming it get an agent. A mismatch here is a silent call.")
    else:
        print("Dispatch is automatic (DANA_AGENT_NAME unset): the worker joins every")
        print("room in the project. Fine for one agent, wrong the moment there are two.")
    print()
    print("Then, with `lk` authenticated against the project:")
    print()
    print(f"  lk sip inbound create {trunk_path.as_posix()}")
    if args.trunk_id:
        print(f"  lk sip dispatch-rule create {rule_path.as_posix()}")
    else:
        print("  # take the ST_... id it prints, then:")
        print(f"  python scripts/sip_config.py --trunk-id ST_...")
        print(f"  lk sip dispatch-rule create {rule_path.as_posix()}")
    print()
    print("On the Twilio side, the number above must be attached to an Elastic SIP")
    print("Trunk whose Origination URI points at this project, with the codec left")
    print("on PCMU (G.711 u-law) -- the pipeline is ulaw_8000 end to end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
