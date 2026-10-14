#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "quality"


def run(name: str, command: list[str]) -> dict[str, object]:
    started = time.perf_counter()
    proc = subprocess.run(command, cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True)
    (REPORT_DIR / f"{name}.txt").write_text(
        f"$ {' '.join(command)}\n\n{proc.stdout}\n{proc.stderr}", encoding="utf-8"
    )
    return {"name": name, "command": command, "returncode": proc.returncode,
            "duration_seconds": round(time.perf_counter() - started, 3)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage-fail-under", type=float, default=70.0)
    args = parser.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    checks = [
        run("ruff", [sys.executable, "-m", "ruff", "check", "--select", "E9,F63,F7,F82", "backend", "tests"]),
        run("mypy", [sys.executable, "-m", "mypy", "backend/config.py", "backend/rag", "backend/feedback.py"]),
        run("pytest", [sys.executable, "-m", "pytest", "-q", "--cov=backend.rag", "--cov=backend.config",
                       "--cov=backend.feedback", "--cov-report=term-missing",
                       f"--cov-fail-under={args.coverage_fail_under}",
                       "--junitxml=reports/quality/junit.xml", "--cov-report=xml:reports/quality/coverage.xml"]),
        run("rag-eval", [sys.executable, "-m", "backend.evaluation.eval_rag"]),
    ]
    result = {
        "run_id": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "created_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "status": "passed" if all(c["returncode"] == 0 for c in checks) else "failed",
        "checks": checks,
        "pyproject_sha256": hashlib.sha256((ROOT / "pyproject.toml").read_bytes()).hexdigest(),
    }
    (REPORT_DIR / "latest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 质量检查摘要", "", f"- 状态: **{result['status']}**", "", "| 检查 | 状态 | 耗时(s) |", "|---|---:|---:|"]
    lines += [f"| {c['name']} | {'通过' if c['returncode'] == 0 else '失败'} | {c['duration_seconds']} |" for c in checks]
    (REPORT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

