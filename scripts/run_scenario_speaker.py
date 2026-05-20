#!/usr/bin/env python3
"""Run scenario-file steps against ai-server and play responses on a local speaker.

Uses the same scenario schema as ``validate_backend_live.py`` (JSON or Markdown
with a fenced ```json block). Each step sends ``stt_result`` over ``/ws/chat`` and
speaks ``filler`` / ``response`` / ``identity_check`` messages via gTTS + ALSA.

Typical usage on Jetson (with Jabra):

    cd ~/Capstone-Project/ai-server
    PYTHONPATH=../local_agent/src:scripts python3 scripts/run_scenario_speaker.py \\
      --scenario-file scripts/odiss_engine_call_trace_scenarios.json \\
      --scenario engine_ce_smalltalk_no_med_contamination \\
      --backend-url http://192.168.0.12:8000

List scenario ids:

    python3 scripts/run_scenario_speaker.py \\
      --scenario-file scripts/odiss_engine_call_trace_scenarios.json --list
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import websockets

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from scenario_loader import load_scenarios_from_file  # noqa: E402

logger = logging.getLogger(__name__)

SPOKEN_TYPES = frozenset({"filler", "response", "identity_check", "reminder", "ocr_processed"})


def resolve_local_agent_root() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    for candidate in (repo_root / "local_agent", Path.home() / "local_agent"):
        if (candidate / "src" / "config_loader.py").exists():
            return candidate
    raise FileNotFoundError("local_agent package not found (expected ~/local_agent or ../local_agent)")


def build_tts(*, stub: bool):
    local_agent_root = resolve_local_agent_root()
    if str(local_agent_root) not in sys.path:
        sys.path.insert(0, str(local_agent_root))
    os.chdir(local_agent_root)

    from src.config_loader import load_config
    from src.edge_node.tts import GTTSEngine, StubTTS, TTSPriority
    from src.home_environment.speaker import LocalSpeaker

    if stub:
        return StubTTS(), TTSPriority

    cfg = load_config()
    audio_cfg = cfg.get("audio", {})
    tts_cfg = cfg.get("tts", {})
    speaker = LocalSpeaker(
        device=audio_cfg.get("output_device", "default"),
        sample_rate=tts_cfg.get("sample_rate", audio_cfg.get("sample_rate", 22050)),
        channels=tts_cfg.get("channels", audio_cfg.get("channels", 1)),
    )
    tts = GTTSEngine(
        speaker=speaker,
        lang=tts_cfg.get("lang", "ko"),
        tld=tts_cfg.get("tld", "co.kr"),
        slow=tts_cfg.get("slow", False),
        sample_rate=tts_cfg.get("sample_rate", 22050),
        channels=tts_cfg.get("channels", 1),
    )
    return tts, TTSPriority


def spoken_text(message: dict[str, Any]) -> str:
    return (
        message.get("response_text")
        or message.get("text")
        or message.get("message")
        or ""
    ).strip()


async def speak_message(tts: Any, priority_cls: Any, message: dict[str, Any]) -> bool:
    msg_type = message.get("type", "")
    text = spoken_text(message)
    if not text or msg_type not in SPOKEN_TYPES:
        return False
    if msg_type == "filler":
        await tts.speak(text, priority_cls.NORMAL)
    else:
        if message.get("requires_tts", True):
            await tts.speak(text, priority_cls.HIGH)
    return True


async def run_step_over_ws(
    websocket: Any,
    *,
    speaker_id: str,
    text: str,
    tts: Any,
    priority_cls: Any,
    recv_limit: int = 8,
    recv_timeout: float = 120.0,
) -> list[dict[str, Any]]:
    await websocket.send(
        json.dumps(
            {"type": "stt_result", "text": text, "speaker_id": speaker_id},
            ensure_ascii=False,
        )
    )
    messages: list[dict[str, Any]] = []
    played = 0
    for _ in range(recv_limit):
        raw = await asyncio.wait_for(websocket.recv(), timeout=recv_timeout)
        message = json.loads(raw)
        messages.append(message)
        if await speak_message(tts, priority_cls, message):
            played += 1
        if message.get("type") in {"response", "error", "identity_check"}:
            break
    return messages


async def run_scenarios(
    *,
    scenarios: list[dict[str, Any]],
    backend_url: str,
    tts: Any,
    priority_cls: Any,
    pause_sec: float,
    run_id: str,
) -> list[dict[str, Any]]:
    ws_url = backend_url.replace("http://", "ws://").replace("https://", "wss://")
    if not ws_url.endswith("/ws/chat"):
        ws_url = ws_url.rstrip("/") + "/ws/chat"

    summaries: list[dict[str, Any]] = []

    async with websockets.connect(ws_url, open_timeout=15) as websocket:
        for scenario in scenarios:
            speaker_id = f"{scenario['speaker_id']}_{run_id[:8]}"
            print(f"\n=== scenario={scenario['id']} speaker_id={speaker_id} ===")
            for index, step in enumerate(scenario["steps"], start=1):
                step_id = step.get("id", f"step_{index}")
                text = step["text"]
                print(f"\n--- step {index}: {step_id} ---")
                print(f"USER: {text}")
                started = time.perf_counter()
                try:
                    messages = await run_step_over_ws(
                        websocket,
                        speaker_id=speaker_id,
                        text=text,
                        tts=tts,
                        priority_cls=priority_cls,
                    )
                    status = "ok"
                    error = ""
                except Exception as exc:  # noqa: BLE001
                    messages = []
                    status = "error"
                    error = repr(exc)
                    logger.exception("step failed")
                elapsed_ms = (time.perf_counter() - started) * 1000
                final = next(
                    (m for m in messages if m.get("type") in {"response", "identity_check"}),
                    {},
                )
                answer = spoken_text(final)
                print(f"ASSISTANT ({elapsed_ms:.0f}ms): {answer[:200]}")
                summaries.append(
                    {
                        "scenario_id": scenario["id"],
                        "step_id": step_id,
                        "status": status,
                        "elapsed_ms": round(elapsed_ms, 1),
                        "answer_preview": answer[:300],
                        "message_types": [m.get("type") for m in messages],
                        "error": error,
                    }
                )
                if pause_sec > 0:
                    await asyncio.sleep(pause_sec)

    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play scenario steps through ai-server WS + speaker.")
    parser.add_argument(
        "--scenario-file",
        type=Path,
        required=True,
        help="JSON scenario suite or Markdown note with fenced json.",
    )
    parser.add_argument("--scenario", help="Run only this scenario id.")
    parser.add_argument("--list", action="store_true", help="List scenario ids and exit.")
    parser.add_argument("--backend-url", default="http://192.168.0.12:8000")
    parser.add_argument("--pause-sec", type=float, default=2.0, help="Pause between steps.")
    parser.add_argument("--stub-tts", action="store_true", help="Log only, no speaker output.")
    parser.add_argument("--output", type=Path, help="Write JSON summary to this path.")
    return parser.parse_args()


async def async_main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    scenarios = load_scenarios_from_file(args.scenario_file)
    if args.list:
        for item in scenarios:
            print(f"{item['id']}\t{len(item['steps'])} steps\trunner={item.get('runner')}")
        return 0

    if args.scenario:
        scenarios = [s for s in scenarios if s["id"] == args.scenario]
        if not scenarios:
            raise SystemExit(f"Scenario not found: {args.scenario}")

    tts, priority_cls = build_tts(stub=args.stub_tts)
    run_id = uuid.uuid4().hex[:8]
    print(f"backend={args.backend_url} scenarios={len(scenarios)} run_id={run_id} stub_tts={args.stub_tts}")

    summaries = await run_scenarios(
        scenarios=scenarios,
        backend_url=args.backend_url.rstrip("/"),
        tts=tts,
        priority_cls=priority_cls,
        pause_sec=args.pause_sec,
        run_id=run_id,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote {args.output}")

    errors = [row for row in summaries if row["status"] == "error"]
    print(f"\nDone: {len(summaries)} steps, {len(errors)} errors")
    return 1 if errors else 0


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
