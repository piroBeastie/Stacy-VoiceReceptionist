"""Time the candidate models from the worker, where the network distance is production's.

Dispatches a job into a `bench-` room. The entrypoint returns before any session
exists, runs :mod:`stacy.bench.llm_latency`, and logs one JSON line per model.
Read them with `lk agent logs`, or with the recorder you already have running.

    python scripts/bench_models.py [--wait-s 180]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402
from livekit import api  # noqa: E402

from stacy.bench import BENCH_PREFIX  # noqa: E402


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wait-s", type=int, default=180, help="How long to leave the room open")
    ap.add_argument("--models", default="", help="Comma-separated vendor/model, else the module's list")
    ap.add_argument("--rounds", type=int, default=0)
    ap.add_argument("--gap-s", type=float, default=0.0, help="Pause between calls; LiveKit Inference 429s on a wide burst")
    args = ap.parse_args()

    load_dotenv()
    agent_name = os.getenv("STACY_AGENT_NAME", "").strip() or "stacy"
    lk = api.LiveKitAPI(
        os.environ["LIVEKIT_URL"].replace("wss://", "https://"),
        os.environ["LIVEKIT_API_KEY"],
        os.environ["LIVEKIT_API_SECRET"],
    )
    room = f"{BENCH_PREFIX}{int(time.time())}"
    try:
        asked: dict = {}
        if args.models:
            asked["models"] = [m.strip() for m in args.models.split(",") if m.strip()]
        if args.rounds:
            asked["rounds"] = args.rounds
        if args.gap_s:
            asked["gap_s"] = args.gap_s
        await lk.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=agent_name, room=room, metadata=json.dumps(asked) if asked else ""
            )
        )
        print(f"dispatched {agent_name} into {room}; the numbers go to the worker log", flush=True)
        await asyncio.sleep(args.wait_s)
        try:
            await lk.room.delete_room(api.DeleteRoomRequest(room=room))
        except Exception:  # noqa: BLE001, S110 - an empty room closes on its own
            pass
    finally:
        await lk.aclose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
