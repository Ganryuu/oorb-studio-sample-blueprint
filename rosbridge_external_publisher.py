#!/usr/bin/env python3
"""
OORB Studio ROS Bridge - External Publisher (manual)

Use this on any external machine running ROS (robot, VM, laptop) to publish ROS messages into
an OORB Studio workspace through the Studio rosbridge-proxy.

Prereqs:
  - Python 3.10+
  - pip install websockets (prefer a venv)

How to get the URL:
  - In the Studio browser: DevTools -> Network -> WS -> click "rosbridge-proxy" -> copy Request URL
  - It looks like:
      wss://api.oorb.io/api/v1/projects/<project_id>/container/rosbridge-proxy?token=<JWT>

Security:
  - The token in the URL is sensitive. Do not paste it into tickets, docs, or commits.

Then run the following example via oorb studio alongside a publisher node:
Example:
  export ROSBRIDGE_WSS='wss://api.oorb.io/api/v1/projects/<project_id>/container/rosbridge-proxy?token=<JWT>'
  python3 rosbridge_external_publisher.py --topic /cmd_vel --type geometry_msgs/Twist --rate 10 \
    --msg '{"linear":{"x":0.5,"y":0.0,"z":0.0},"angular":{"x":0.0,"y":0.0,"z":-0.5}}'

Notes:
  - Some deployments enforce Origin checks; this client sets Origin=https://oorb.io by default.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import time

import websockets


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _ws_connect_kwargs(origin: str) -> dict:
    sig = inspect.signature(websockets.connect)
    if "additional_headers" in sig.parameters:
        return {"additional_headers": {"Origin": origin}}
    if "extra_headers" in sig.parameters:
        return {"extra_headers": {"Origin": origin}}
    return {}


async def _receiver(ws) -> None:
    async for raw in ws:
        try:
            data = json.loads(raw)
            _log(f"RX {data}")
        except Exception:
            _log(f"RX (non-JSON) {raw}")


async def _run(args) -> None:
    _log(f"Connecting to {args.url}")
    kwargs = _ws_connect_kwargs(args.origin)
    async with websockets.connect(
        args.url,
        ping_interval=20,
        ping_timeout=20,
        close_timeout=5,
        **kwargs,
    ) as ws:
        _log("Connected")

        if args.check_rosapi:
            await ws.send(json.dumps({"op": "call_service", "service": "/rosapi/get_time", "args": {}}))
            _log("Sent /rosapi/get_time")

        recv_task = asyncio.create_task(_receiver(ws)) if args.print_rx else None

        await ws.send(json.dumps({"op": "advertise", "topic": args.topic, "type": args.msg_type}))
        _log(f"Advertised {args.topic} ({args.msg_type})")

        payload = json.loads(args.msg)
        period = 1.0 / max(args.rate, 0.001)
        start = time.time()
        sent = 0

        try:
            while True:
                await ws.send(json.dumps({"op": "publish", "topic": args.topic, "msg": payload}))
                sent += 1
                if sent % int(max(args.rate, 1)) == 0:
                    _log(f"Published {sent} messages")

                await asyncio.sleep(period)

                if args.duration > 0 and (time.time() - start) >= args.duration:
                    _log("Duration reached, stopping")
                    break
        finally:
            if recv_task is not None:
                recv_task.cancel()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.getenv("ROSBRIDGE_WSS", ""), help="Full rosbridge wss URL")
    parser.add_argument("--origin", default="https://oorb.io", help="Origin header")
    parser.add_argument("--topic", default="/cmd_vel", help="Publish topic")
    parser.add_argument("--type", dest="msg_type", default="geometry_msgs/Twist", help="ROS message type")
    parser.add_argument("--rate", type=float, default=10.0, help="Publish rate (Hz)")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run (0 = forever)")
    parser.add_argument(
        "--msg",
        default='{"linear":{"x":0.5,"y":0.0,"z":0.0},"angular":{"x":0.0,"y":0.0,"z":-0.5}}',
        help="JSON message payload",
    )
    parser.add_argument("--check-rosapi", action="store_true", help="Call /rosapi/get_time on connect")
    parser.add_argument("--print-rx", action="store_true", help="Print incoming rosbridge messages")
    args = parser.parse_args()

    if not args.url:
        raise SystemExit("Missing --url (or set ROSBRIDGE_WSS)")

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
