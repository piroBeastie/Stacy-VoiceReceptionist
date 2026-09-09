"""Check everything a call needs before a caller is on the line.

    python scripts/preflight.py

Voice agents fail expensively: a missing key surfaces as dead air to somebody
who dialled a real business, and you find out from them. Every check below is
something that has to be true before the first call, and all of them are
cheaper to run now than to diagnose then.

FAIL means a call will not work. WARN means it will work and something you
probably want is off -- an unconfigured webhook still answers the phone, it just
quietly breaks the promise the agent makes at the end of every call.

Exits non-zero on any FAIL, so it can gate a deploy.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dana.prompt import PromptBuilder  # noqa: E402
from dana.providers.registry import pipeline_for  # noqa: E402
from dana.tenants import TenantStore  # noqa: E402

OK, WARN, FAIL = "PASS", "WARN", "FAIL"
_SYMBOL = {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL "}

results: list[tuple[str, str, str]] = []


def check(level: str, name: str, detail: str = "") -> None:
    results.append((level, name, detail))


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def check_dependencies() -> None:
    """The SDK and the plugins the configured vendors need."""
    for module, why in (
        ("livekit.agents", "the worker itself"),
        ("livekit.plugins.silero", "VAD"),
        ("livekit.plugins.deepgram", "STT"),
        ("livekit.plugins.openai", "LLM (any OpenAI-compatible host)"),
        ("livekit.plugins.elevenlabs", "TTS"),
        ("livekit.plugins.turn_detector", "semantic endpointing"),
        ("aiohttp", "post-call webhook delivery"),
    ):
        try:
            # find_spec raises rather than returning None when the *parent*
            # package is absent -- which is the ordinary "nothing installed yet"
            # case this whole script exists to report politely.
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            found = False
        if found:
            check(OK, module, why)
        else:
            check(FAIL, module, f"not installed -- needed for {why}. Run: pip install -e .")


def check_env() -> None:
    """Credentials, split by what breaks without them."""
    required = {
        "LIVEKIT_URL": "the worker cannot register",
        "LIVEKIT_API_KEY": "the worker cannot register",
        "LIVEKIT_API_SECRET": "the worker cannot register",
        "DEEPGRAM_API_KEY": "no transcription",
        "LLM_API_KEY": "no responses",
    }
    for name, consequence in required.items():
        check(OK if _env(name) else FAIL, name, "" if _env(name) else consequence)

    if not _env("ELEVENLABS_API_KEY") and not _env("CARTESIA_API_KEY"):
        check(FAIL, "TTS key", "set ELEVENLABS_API_KEY or CARTESIA_API_KEY — no voice otherwise")
    else:
        check(OK, "TTS key", "elevenlabs" if _env("ELEVENLABS_API_KEY") else "cartesia")

    vendor = _env("LLM_VENDOR") or "anthropic"
    check(OK, "LLM", f"{vendor}/{_env('LLM_MODEL') or 'claude-sonnet-5'}")
    if vendor == "anthropic" and _env("LLM_CACHING") in {"0", "false", "no"}:
        check(
            WARN,
            "LLM_CACHING",
            "off — Anthropic does not cache unless asked, so every turn pays for "
            "the full ~5,700-token tenant prompt in latency and money",
        )

    if not _env("LLM_FALLBACK_MODEL"):
        check(
            WARN,
            "LLM_FALLBACK_MODEL",
            "no cascade target — a rate-limited primary is a 4s hole in the call",
        )
    elif not _env("LLM_FALLBACK_API_KEY"):
        check(
            FAIL,
            "LLM_FALLBACK_API_KEY",
            f"{_env('LLM_FALLBACK_MODEL')} is configured but has no key. The fallback "
            "is a different account from the primary and cannot borrow its credential — "
            "it would fail authentication at the one moment it is needed",
        )
    else:
        check(
            OK,
            "LLM_FALLBACK_MODEL",
            f"{_env('LLM_FALLBACK_VENDOR') or 'openai'}/{_env('LLM_FALLBACK_MODEL')}",
        )

    if not (_env("DANA_WEBHOOK_URL") and _env("DANA_WEBHOOK_SECRET")):
        check(
            WARN,
            "post-call webhook",
            "unset — the closing promise that someone gets the caller's details "
            "will not be kept (calls are still archived locally)",
        )
    else:
        check(OK, "post-call webhook", _env("DANA_WEBHOOK_URL"))

    if _env("EVAL_MODEL") and _env("EVAL_MODEL") == _env("LLM_MODEL"):
        check(WARN, "EVAL_MODEL", "same as LLM_MODEL — that is self-marking")


def check_tenants(tenant_dir: str) -> None:
    """Every tenant loads, renders, and has somewhere to send a call."""
    try:
        store = TenantStore(tenant_dir)
    except Exception as exc:  # noqa: BLE001
        check(FAIL, "tenants", f"{tenant_dir}: {exc}")
        return

    tenants = store.all()
    if not tenants:
        check(FAIL, "tenants", f"no *.json in {tenant_dir}")
        return
    check(OK, "tenants", f"{len(tenants)} loaded from {tenant_dir}")

    for tenant in tenants:
        label = f"tenant:{tenant.tenant_id}"
        try:
            prompt = PromptBuilder().render(tenant)
        except Exception as exc:  # noqa: BLE001
            check(FAIL, label, f"prompt render failed: {exc}")
            continue

        cfg = pipeline_for(tenant)
        if not cfg.tts.voice_id:
            check(FAIL, label, "no voice_id — TTS has no voice to speak in")
            continue
        if not tenant.greeting.strip():
            check(FAIL, label, "no greeting — the call opens with silence")
            continue

        notes = [f"{tenant.phone_number}", f"~{len(prompt) // 4} tok"]
        if not tenant.hours:
            notes.append("no hours in the knowledge base")
        if not tenant.services:
            notes.append("no services/pricing")
        level = WARN if len(notes) > 2 else OK
        check(level, label, ", ".join(notes))

    default = _env("DANA_DEFAULT_TENANT")
    if default:
        try:
            store.get(default)
            check(OK, "DANA_DEFAULT_TENANT", f"{default} (console/dev routing)")
        except KeyError:
            check(FAIL, "DANA_DEFAULT_TENANT", f"{default!r} is not a known tenant_id")
    else:
        check(WARN, "DANA_DEFAULT_TENANT", "unset — console and dev mode cannot route a call")


def check_sip(tenant_dir: str) -> None:
    """The generated SIP config, and whether it still matches the tenants.

    The dialed number is the routing key. If the LiveKit trunk carries a number
    no tenant claims, the call connects, the room is created, and the agent
    raises KeyError with a customer on the line -- so a drift between these two
    files is worth catching on the ground.
    """
    trunk_file = Path("deploy/livekit/inbound-trunk.json")
    rule_file = Path("deploy/livekit/dispatch-rule.json")
    if not trunk_file.exists():
        check(WARN, "SIP config", "not generated yet — run scripts/sip_config.py")
        return

    try:
        trunk = json.loads(trunk_file.read_text(encoding="utf-8"))
        configured = set(trunk["trunk"]["numbers"])
    except Exception as exc:  # noqa: BLE001
        check(FAIL, "SIP config", f"{trunk_file}: {exc}")
        return

    try:
        claimed = {t.phone_number for t in TenantStore(tenant_dir).all()}
    except Exception:  # noqa: BLE001
        return  # already reported by check_tenants

    unrouted = configured - claimed
    uncarried = claimed - configured
    if unrouted:
        check(
            FAIL,
            "SIP numbers",
            f"trunk carries {sorted(unrouted)} that no tenant claims — those calls "
            "connect and then crash. Re-run scripts/sip_config.py",
        )
    elif uncarried:
        check(
            WARN,
            "SIP numbers",
            f"tenants claim {sorted(uncarried)} that the trunk does not carry — "
            "those numbers will not reach the agent",
        )
    else:
        check(OK, "SIP numbers", f"{len(configured)} number(s) match the tenants")

    agent_name = _env("DANA_AGENT_NAME")
    named_in_rule = ""
    if rule_file.exists():
        try:
            rule = json.loads(rule_file.read_text(encoding="utf-8"))
            agents = rule.get("room_config", {}).get("agents", [])
            named_in_rule = agents[0].get("agent_name", "") if agents else ""
            if not rule.get("trunk_ids") or not rule["trunk_ids"][0]:
                check(
                    WARN,
                    "dispatch rule",
                    "no trunk_id — run sip_config.py --trunk-id ST_... after creating the trunk",
                )
        except Exception as exc:  # noqa: BLE001
            check(FAIL, "dispatch rule", f"{rule_file}: {exc}")

    if agent_name != named_in_rule:
        check(
            FAIL,
            "agent dispatch",
            f"worker registers as {agent_name or '(automatic)'} but the rule dispatches "
            f"{named_in_rule or '(automatic)'} — the call connects and nobody joins",
        )
    else:
        check(OK, "agent dispatch", agent_name or "automatic (worker joins every room)")


def check_storage() -> None:
    """The directories the call writes to have to exist and be writable."""
    data = _env("DANA_DATA_DIR") or "data"
    traces = _env("DANA_TRACE_DIR") or "data/traces"
    for path in (f"{data}/calendar", f"{data}/messages", f"{data}/calls", traces):
        try:
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            probe = p / ".preflight"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            check(OK, f"writable {path}")
        except Exception as exc:  # noqa: BLE001
            check(FAIL, f"writable {path}", str(exc))


def check_known_gaps() -> None:
    """Things that are deliberately unfinished and should be said out loud."""
    check(
        WARN,
        "SMS provider",
        "LoggingProvider — confirmations are written to the log, not sent",
    )
    check(
        WARN,
        "retention",
        "transcripts, traces and the calendar are plaintext on disk with no expiry",
    )
    check(
        WARN,
        "horizontal scale",
        "CalendarStore's lock is per-process — run one worker, or two can "
        "double-book the same window",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tenant-dir", default=os.getenv("DANA_TENANT_DIR", "src/dana/tenants/data"))
    ap.add_argument("--strict", action="store_true", help="Treat warnings as failures too")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        from dotenv import load_dotenv  # noqa: PLC0415

        load_dotenv()
    except ImportError:
        pass

    check_dependencies()
    check_env()
    check_tenants(args.tenant_dir)
    check_sip(args.tenant_dir)
    check_storage()
    check_known_gaps()

    width = max(len(name) for _, name, _ in results) + 2
    for level, name, detail in results:
        print(f"[{_SYMBOL[level]}] {name:<{width}} {detail}")

    fails = sum(1 for level, _, _ in results if level == FAIL)
    warns = sum(1 for level, _, _ in results if level == WARN)
    print()
    print(f"{fails} failing, {warns} warnings, {len(results)} checks")
    if fails:
        print("\nA call will not work until the FAIL rows are fixed.")
    return 1 if fails or (args.strict and warns) else 0


if __name__ == "__main__":
    raise SystemExit(main())
