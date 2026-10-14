"""分块策略 A/B 基准。

对比三种分块方式在检索指标上的差异：

* ``fixed``             —— 原实现：按字符长度硬切（step = size - overlap）。
* ``structure``         —— 按文档章节标题切分，相邻小节合并到不超过 size，
                            单节超长时退化到句子边界。
* ``structure_context`` —— 在 structure 基础上，每块前面加上文档标题作为上下文提示。

只在文档长度超过 chunk_size 时这三者才会产生差异，因此本脚本对
200 / 300 / 400 三档 chunk_size 分别测量。

用法::

    python -m backend.evaluation.bench_chunking
    python -m backend.evaluation.bench_chunking --sizes 200 400
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.rag.embeddings import create_embedding_provider  # noqa: E402
from backend.rag.service import (  # noqa: E402
    CHUNK_STRATEGIES,
    HybridRagService,
    load_documents,
)

CASES = {
    "v1": ROOT / "backend" / "evaluation" / "rag_cases.json",
    "v2": ROOT / "backend" / "evaluation" / "rag_cases_v2.json",
}
PROVIDERS = [("hash-v1", False), ("BGE", True)]
MODEL = "BAAI/bge-small-zh-v1.5"


def evaluate(service: HybridRagService, cases: dict) -> dict:
    recall_at_1 = recall_at_5 = 0
    reciprocal_ranks: list[float] = []
    for case in cases["retrieval"]:
        response = service.search(case["query"], top_k=5)
        rank = None
        for position, item in enumerate(response["results"], start=1):
            if item["source_id"] == case["expected_source_id"]:
                rank = position
                break
        recall_at_1 += int(rank == 1)
        recall_at_5 += int(rank is not None and rank <= 5)
        reciprocal_ranks.append(0.0 if rank is None else 1 / rank)
    n = len(cases["retrieval"])
    return {"recall_at_1": recall_at_1 / n, "recall_at_5": recall_at_5 / n,
            "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 300, 400])
    args = parser.parse_args()

    documents = load_documents()
    loaded = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in CASES.items()}

    # embedding 实例跨配置复用，避免重复加载模型
    providers = {label: create_embedding_provider(MODEL, use_real) for label, use_real in PROVIDERS}

    rows = []
    for label, _ in PROVIDERS:
        for size in args.sizes:
            for strategy in CHUNK_STRATEGIES:
                service = HybridRagService(
                    providers[label], chunk_size=size, chunk_overlap=64, chunk_strategy=strategy
                )
                info = service.index(documents)
                for set_name, cases in loaded.items():
                    metrics = evaluate(service, cases)
                    rows.append({"provider": label, "chunk_size": size, "strategy": strategy,
                                 "case_set": set_name, "chunks": info["chunks"], **metrics})

    print(f"{'provider':9s} {'size':>5s} {'策略':19s} {'块数':>5s} {'v1 R@1':>8s} {'v2 R@1':>8s} {'v2 R@5':>8s} {'v2 MRR':>8s}")
    print("-" * 84)
    for label, _ in PROVIDERS:
        for size in args.sizes:
            for strategy in CHUNK_STRATEGIES:
                subset = [r for r in rows if r["provider"] == label and r["chunk_size"] == size
                          and r["strategy"] == strategy]
                v1 = next(r for r in subset if r["case_set"] == "v1")
                v2 = next(r for r in subset if r["case_set"] == "v2")
                print(f"{label:9s} {size:5d} {strategy:19s} {v2['chunks']:5d} "
                      f"{v1['recall_at_1']:8.3f} {v2['recall_at_1']:8.3f} "
                      f"{v2['recall_at_5']:8.3f} {v2['mrr']:8.3f}")

    output = ROOT / "reports" / "bench" / "chunking.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"results": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print("已写入:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
