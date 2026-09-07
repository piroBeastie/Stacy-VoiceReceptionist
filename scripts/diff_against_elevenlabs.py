"""Diff the rendered prompt against the live ElevenLabs agent.

Parity check for the migration window, while both stacks are running. Any
change made in the ElevenLabs dashboard shows up here as a drift the port has
not absorbed yet.

    ELEVENLABS_API_KEY=... python scripts/diff_against_elevenlabs.py

Exits 1 on any difference.

NOTE: the port has now intentionally diverged in two sections. `# Knowledge`
carries the business facts the platform kept in a per-agent RAG document, and
`# Tools` describes five tools the platform agent does not have. Expect both as
drift, and read this as an advisory diff rather than a CI gate until the
migration window closes.
"""

from __future__ import annotations

import difflib
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dana.prompt import PromptBuilder  # noqa: E402
from dana.tenants import TenantStore  # noqa: E402

AGENT_ID = "agent_9901m0z6x9kse2z91paeeyrzb36s"
BRANCH_ID = "agtbrch_1801m0z6xaf2f7c8vmdntgj8snqn"
API = "https://api.elevenlabs.io/v1/convai/agents"


def fetch_remote_prompt(api_key: str) -> str:
    """Pull main's current system prompt."""
    import json

    req = urllib.request.Request(
        f"{API}/{AGENT_ID}?branch_id={BRANCH_ID}", headers={"xi-api-key": api_key}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.load(resp)
    return body["conversation_config"]["agent"]["prompt"]["prompt"]


def main() -> int:
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("ELEVENLABS_API_KEY not set", file=sys.stderr)
        return 2

    store = TenantStore(os.getenv("DANA_TENANT_DIR", "src/dana/tenants/data"))
    local = PromptBuilder().render(store.get("millbrook"))
    remote = fetch_remote_prompt(api_key)

    diff = list(
        difflib.unified_diff(
            remote.splitlines(),
            local.splitlines(),
            fromfile="elevenlabs/main",
            tofile="dana/rendered",
            lineterm="",
            n=1,
        )
    )
    if not diff:
        print("identical")
        return 0

    print("\n".join(diff))
    added = sum(1 for d in diff if d.startswith("+") and not d.startswith("+++"))
    removed = sum(1 for d in diff if d.startswith("-") and not d.startswith("---"))
    print(f"\n{removed} lines only on ElevenLabs, {added} lines only local", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
