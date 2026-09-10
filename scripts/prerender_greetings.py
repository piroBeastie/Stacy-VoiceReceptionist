"""Render every tenant's greeting to the cache, before any call arrives.

    python scripts/prerender_greetings.py

The worker renders a greeting on first use and caches it, so this is not
required -- but without it the *first* caller after a deploy pays the synthesis,
and the first caller is exactly who you least want to make wait. Run it as a
deploy step.

Safe to run repeatedly: an already-rendered greeting is left alone, and editing
a tenant's greeting invalidates its own entry because the text is part of the
cache key.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stacy.agent import greeting  # noqa: E402
from stacy.providers.registry import build_tts, pipeline_for, register_plugins  # noqa: E402
from stacy.tenants import TenantStore  # noqa: E402


async def run(tenant_dir: str, cache_dir: str) -> int:
    # The plugins reach for the worker's shared HTTP session, which only exists
    # inside a job. Outside one -- here -- it has to be opened explicitly, or
    # every synthesis fails with "http session outside of a job context".
    from livekit.agents.utils import http_context  # noqa: PLC0415

    tenants = TenantStore(tenant_dir).all()
    if not tenants:
        print(f"no tenants in {tenant_dir}", file=sys.stderr)
        return 1

    rendered = 0
    async with http_context.open():
        for tenant in tenants:
            cfg = pipeline_for(tenant).tts
            audio = await greeting.prepare(tenant, cfg, cache_dir, build_tts)
            if audio is None:
                print(f"  {tenant.tenant_id}: FAILED — it will be synthesised live")
                continue
            frames = total = 0
            async for frame in audio:
                frames += 1
                total += len(frame.data)
            secs = total / max(1, frame.sample_rate * frame.num_channels * 2)
            print(f"  {tenant.tenant_id}: {secs:.1f}s of speech, {frames} frames")
            rendered += 1

    print(f"\n{rendered}/{len(tenants)} greetings ready in {cache_dir}")
    return 0 if rendered == len(tenants) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tenant-dir", default=os.getenv("STACY_TENANT_DIR", "src/stacy/tenants/data"))
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        from dotenv import load_dotenv  # noqa: PLC0415

        load_dotenv()
    except ImportError:
        pass

    cache = args.cache_dir or f"{os.getenv('STACY_DATA_DIR', 'data')}/greetings"
    register_plugins()
    return asyncio.run(run(args.tenant_dir, cache))


if __name__ == "__main__":
    raise SystemExit(main())
