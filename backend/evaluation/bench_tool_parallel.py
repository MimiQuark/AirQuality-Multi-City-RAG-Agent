"""工具并发执行基准：顺序执行 vs 线程池并发。

背景：Agent 的系统 Prompt 与节点注释都写着"并行执行所有工具"，但原实现是
顺序 for 循环。改为并发后，本脚本用真实工具组合量化实际收益。

为什么收益不是线性的：这些工具是 CPU 密集型（pandas / sklearn），
asyncio.to_thread 走线程池，仍受 CPython GIL 约束，只有释放 GIL 的
数值计算段能真正重叠。因此只有在**多个重工具**同批调用时才看得出差别。

为什么不用进程池：工具依赖模块级 DataFrame 缓存（tools.implementations._df_cache），
多进程会各自重新加载数据与拟合模型，启动成本反而超过收益。

用法::

    python -m backend.evaluation.bench_tool_parallel
    python -m backend.evaluation.bench_tool_parallel --repeat 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.tools.implementations import execute_tool_call, execute_tool_calls  # noqa: E402

PARAMS: dict[str, dict] = {
    "get_overview": {"city": "beijing"},
    "get_forecast": {"city": "beijing", "pollutant": "PM25"},
    "get_correlation": {"city": "beijing"},
    "get_alerts": {"city": "beijing"},
    "get_comprehensive_alerts": {"city": "beijing"},
    "get_season_analysis": {"city": "beijing", "pollutant": "PM25"},
    "get_cities_comparison": {},
    "train_models": {"city": "beijing"},
    "get_realtime": {"city": "beijing"},
    "query_policy": {"query": "PM2.5 标准限值"},
}


def calls(*names: str) -> list[dict]:
    return [{"tool": name, "params": dict(PARAMS[name])} for name in names]


SCENARIOS: list[tuple[str, list[dict]]] = [
    ("A 实时+预测（评测集多跳模式）", calls("get_realtime", "get_forecast")),
    ("B 政策+综合预警（规则版政策分支）", calls("query_policy", "get_comprehensive_alerts")),
    ("C 预测+综合预警+模型训练（三个重工具）",
     calls("get_forecast", "get_comprehensive_alerts", "train_models")),
    ("D 全部 10 个工具", calls(*PARAMS.keys())),
]


def time_serial(batch: list[dict]) -> float:
    started = time.perf_counter()
    for call in batch:
        execute_tool_call(call, "beijing")
    return time.perf_counter() - started


def time_parallel(batch: list[dict]) -> float:
    started = time.perf_counter()
    asyncio.run(execute_tool_calls(batch, "beijing"))
    return time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=3, help="每组重复次数，取中位数")
    args = parser.parse_args()

    print("预热：首次调用包含数据与模型冷加载")
    for name in ("get_comprehensive_alerts", "train_models", "get_forecast"):
        started = time.perf_counter()
        execute_tool_call({"tool": name, "params": dict(PARAMS[name])}, "beijing")
        print(f"  {name:26s} {time.perf_counter() - started:6.3f}s")
    print()

    header = f"{'场景':38s} {'工具数':>5s} {'顺序(s)':>9s} {'并发(s)':>9s} {'加速':>7s} {'省下(s)':>9s}"
    print(header)
    print("-" * len(header.encode("gbk", errors="ignore")))

    rows = []
    for label, batch in SCENARIOS:
        serial_times, parallel_times = [], []
        for _ in range(args.repeat):
            serial_times.append(time_serial(batch))
            parallel_times.append(time_parallel(batch))
        serial_median = statistics.median(serial_times)
        parallel_median = statistics.median(parallel_times)
        speedup = serial_median / parallel_median if parallel_median else 0.0
        rows.append({
            "scenario": label,
            "tools": [c["tool"] for c in batch],
            "n_tools": len(batch),
            "serial_median_s": round(serial_median, 3),
            "parallel_median_s": round(parallel_median, 3),
            "speedup": round(speedup, 2),
            "saved_s": round(serial_median - parallel_median, 3),
        })
        print(f"{label:38s} {len(batch):5d} {serial_median:9.3f} {parallel_median:9.3f} "
              f"{speedup:6.2f}x {serial_median - parallel_median:9.3f}")

    output = ROOT / "reports" / "bench" / "tool_parallel.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "repeat": args.repeat,
        "note": "工具为 CPU 密集型，线程池并发受 GIL 限制；收益率取决于同批重工具的个数",
        "results": rows,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print("已写入:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
