"""Feedback capture for bad-case mining and regression datasets."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FEEDBACK_PATH = Path(__file__).parent / "data" / "feedback.jsonl"


def record_feedback(payload: dict[str, Any]) -> dict[str, Any]:
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    item = {"created_at": datetime.now(UTC).isoformat(), **payload}
    with FEEDBACK_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return item
