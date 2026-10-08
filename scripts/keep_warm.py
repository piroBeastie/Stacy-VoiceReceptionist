"""Keep the worker from scaling to zero while people are calling.

    python scripts/keep_warm.py [--every-s 240] [--hours 3]

On LiveKit Cloud's free plan an idle agent shuts down, takes 20-30s to wake,
and the call that wakes it rings out -- then the first answered call is the
slowest of the day, with an empty greeting cache and a cold prompt cache. This
dispatches the agent into an empty `keepwarm-` room every few minutes. The
entrypoint returns at once for those rooms, so a ping costs a few seconds of
agent time and never touches a caller.

It is a stopgap for testing and business hours. The lasting fix is a plan that
keeps production agents warm, or a self-hosted worker.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from contextlib import suppress
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402
from livekit import api  # noqa: E402

from stacy.agent.entrypoint import KEEP_WARM_PREFIX  # noqa: E402


async def ping(lk: api.LiveKitAPI, agent_name: str) -> str:
    room = f"{KEEP_WARM_PREFIX}{int(time.time())}"
    await lk.agent_dispatch.create_dispatch(
        api.CreateAgentDispatchRequest(agent_name=agent_name, room=room)
    )
    await asyncio.sleep(15)
    with suppress(Exception):  # an empty room closes on its own anyway
        await lk.room.delete_room(api.DeleteRoomRequest(room=room))
    return room


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--every-s", type=int, default=240, help="Seconds between pings")
    ap.add_argument("--hours", type=float, default=3.0, help="How long to keep it awake")
    args = ap.parse_args()

    load_dotenv()
    agent_name = os.getenv("STACY_AGENT_NAME", "").strip() or "stacy"
    lk = api.LiveKitAPI(
        os.environ["LIVEKIT_URL"].replace("wss://", "https://"),
        os.environ["LIVEKIT_API_KEY"],
        os.environ["LIVEKIT_API_SECRET"],
    )
    deadline = time.monotonic() + args.hours * 3600
    try:
        while time.monotonic() < deadline:
            try:
                room = await ping(lk, agent_name)
                print(f"{time.strftime('%H:%M:%S')} pinged {agent_name} in {room}", flush=True)
            except Exception as exc:  # noqa: BLE001 - keep pinging through a blip
                print(f"{time.strftime('%H:%M:%S')} ping failed: {exc}", flush=True)
            await asyncio.sleep(max(1, args.every_s - 15))
    finally:
        await lk.aclose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
