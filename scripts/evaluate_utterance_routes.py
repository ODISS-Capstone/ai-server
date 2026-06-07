"""Dry-run ODISS utterance route contracts against the deterministic router."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.assistant_intent_classifier import AssistantIntentClassifier  # noqa: E402


RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "emergency": 3}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default="tests/fixtures/odiss_utterance_dataset.csv")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--json-report", default="")
    args = parser.parse_args()

    fixture = Path(args.fixture)
    if not fixture.is_absolute():
        fixture = ROOT / fixture
    classifier = AssistantIntentClassifier()
    rows = _read_rows(fixture)
    mismatches: list[dict[str, str]] = []
    ignored_count = 0
    unclear_count = 0
    safety_false_negative_count = 0

    for index, row in enumerate(rows, start=2):
        text = row["utterance"]
        active_flow = row.get("active_flow_required") or "none"
        actual = classifier.classify(
            text,
            active_flow=active_flow,
            active_session=True,
            odiss_directed=True,
        )
        expected = {
            "route_label": row["route_label_expected"],
            "engine_scope": row["engine_scope_expected"],
            "risk_level": row["risk_level_expected"],
            "fast_path": row["fast_path_expected"],
            "ui_action": row["ui_action_expected"],
            "db_write_expected": row["db_write_expected"],
        }
        got = {
            "route_label": actual.route_label,
            "engine_scope": actual.engine_scope,
            "risk_level": actual.risk_level,
            "fast_path": actual.fast_path,
            "ui_action": actual.ui_action,
            "db_write_expected": "true" if actual.db_write_expected else "false",
        }
        if actual.engine_scope == "ignored":
            ignored_count += 1
        if actual.route_label == "unclear":
            unclear_count += 1
        if RISK_ORDER.get(got["risk_level"], 0) < RISK_ORDER.get(expected["risk_level"], 0):
            if expected["risk_level"] in {"high", "emergency"}:
                safety_false_negative_count += 1
        for key, expected_value in expected.items():
            if expected_value != got[key]:
                mismatches.append(
                    {
                        "line": str(index),
                        "category": row["category"],
                        "utterance": text,
                        "field": key,
                        "expected": expected_value,
                        "actual": got[key],
                        "active_flow": active_flow,
                        "route_reason": actual.route_reason,
                    }
                )

    summary = {
        "total": len(rows),
        "matched": len(rows) - len({m["line"] for m in mismatches}),
        "mismatch_count": len(mismatches),
        "ignored_count": ignored_count,
        "unclear_count": unclear_count,
        "safety_false_negative_count": safety_false_negative_count,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if mismatches:
        print("\nFirst mismatches:")
        for item in mismatches[:20]:
            print(json.dumps(item, ensure_ascii=False))
    if args.json_report:
        report_path = Path(args.json_report)
        if not report_path.is_absolute():
            report_path = ROOT / report_path
        report_path.write_text(
            json.dumps({"summary": summary, "mismatches": mismatches}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0 if args.report_only or not mismatches else 1


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = []
        for raw in reader:
            rows.append(
                {
                    "category": raw[0],
                    "utterance": raw[1],
                    "intent": raw[2],
                    "risk": raw[3],
                    "source_engine": raw[4],
                    "route_label_expected": raw[header.index("route_label_expected")],
                    "engine_scope_expected": raw[header.index("engine_scope_expected")],
                    "risk_level_expected": raw[header.index("risk_level_expected")],
                    "fast_path_expected": raw[header.index("fast_path_expected")],
                    "ui_action_expected": raw[header.index("ui_action_expected")],
                    "db_write_expected": raw[header.index("db_write_expected")],
                    "active_flow_required": raw[header.index("active_flow_required")],
                }
            )
        return rows


if __name__ == "__main__":
    raise SystemExit(main())
