"""Evaluate hybrid RAG retrieval and refusal behaviour, persisting every result.

两套评测集：

* ``rag_cases.json``（v1 简单集）—— 题面原样包含文档标题，用于回归护栏。
  注意：该集中 100% 的 query 都含有目标文档标题原文，而打分公式里标题精确匹配
  权重为 0.60，等于直接给出答案，因此任何 embedding 都能拿到 Recall@1=1.000。
  它只能当"没退化"的护栏，不能当检索质量证据。
* ``rag_cases_v2.json``（v2 困难集，默认）—— 口语化提问（不含标题关键词）
  加平衡拒答集（24 条超纲应拒 + 24 条领域内应答），用于真实衡量检索与拒答能力。

用法::

    python -m backend.evaluation.eval_rag                    # 默认集 + 默认 embedding
    python -m backend.evaluation.eval_rag --cases v1
    python -m backend.evaluation.eval_rag --all              # 2 套集 x 2 种 embedding 对比表
    python -m backend.evaluation.eval_rag --sweep            # 拒答阈值扫描
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.config import settings
from backend.rag import HybridRagService, create_embedding_provider

ROOT = Path(__file__).resolve().parents[2]
CASE_FILES = {
    "v1": ROOT / "backend" / "evaluation" / "rag_cases.json",
    "v2": ROOT / "backend" / "evaluation" / "rag_cases_v2.json",
}

# 回归护栏阈值：设在各 provider 当前实测值下方约 5 个点，用于拦截退化。
# 绝对水平不是"能力上限"，只代表"不许比这更差"。
GATES = {
    # 离线 hash 只设检索护栏：它的相似度尺度无法区分"领域内口语化提问"与"领域外问题"
    # （锚点得分重叠，不存在可行阈值），对拒答设护栏等于设一个永远为 0.5 的假指标。
    "hash-v1": {"recall_at_5": 0.85, "mrr": 0.60},
    # 语义模型才同时守检索与拒答。需要在 CI 上跑这一档时，装好 sentence-transformers
    # 后设置 USE_REAL_EMBEDDINGS=1 再执行本脚本。
    "_real": {"recall_at_5": 0.90, "mrr": 0.80, "refusal_accuracy": 0.85},
}


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def load_cases(tag: str) -> dict:
    return json.loads(CASE_FILES[tag].read_text(encoding="utf-8"))


def build_service(use_real: bool, model_name: str) -> HybridRagService:
    service = HybridRagService(
        create_embedding_provider(model_name, use_real),
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        min_score=settings.rag_min_score,
        refusal_threshold=settings.rag_refusal_threshold,
    )
    service.index(__import__("backend.rag.service", fromlist=["load_documents"]).load_documents())
    return service


def evaluate(service: HybridRagService, cases: dict) -> dict:
    retrieval_details = []
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
        retrieval_details.append({**case, "rank": rank, "confidence": response["confidence"],
                                  "results": response["results"]})

    refusal_details = []
    should_refuse_total = should_answer_total = 0
    correct_refuse = correct_answer = 0
    for case in cases["refusal"]:
        response = service.search(case["query"], top_k=5)
        correct = bool(response["refused"]) == bool(case["expected_refused"])
        if case["expected_refused"]:
            should_refuse_total += 1
            correct_refuse += int(correct)
        else:
            should_answer_total += 1
            correct_answer += int(correct)
        refusal_details.append({**case, "refused": response["refused"],
                                "confidence": response["confidence"], "correct": correct})

    n = len(cases["retrieval"])
    m = len(cases["refusal"])
    return {
        "embedding": service.embedding.name,
        "refusal_threshold": service.refusal_threshold,
        "min_score": service.min_score,
        "cases": {"retrieval": n, "refusal": m,
                  "refusal_should_refuse": should_refuse_total,
                  "refusal_should_answer": should_answer_total},
        "metrics": {
            "recall_at_1": recall_at_1 / n,
            "recall_at_5": recall_at_5 / n,
            "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
            "refusal_accuracy": (correct_refuse + correct_answer) / m,
            # v1 的拒答集全是负样本，没有正样本可分，此时该指标无定义（None 而不是 0）
            "refusal_accuracy_on_refuse_samples": (
                correct_refuse / should_refuse_total if should_refuse_total else None),
            "answer_accuracy_on_answer_samples": (
                correct_answer / should_answer_total if should_answer_total else None),
        },
        "retrieval_results": retrieval_details,
        "refusal_results": refusal_details,
    }


def gates_for(service: HybridRagService) -> dict:
    return GATES["hash-v1"] if service.embedding.name == "hash-v1" else GATES["_real"]


def run_one(tag: str, use_real: bool, quiet: bool = False) -> dict:
    service = build_service(use_real, settings.embedding_model)
    cases = load_cases(tag)
    result = evaluate(service, cases)
    result["case_set"] = tag
    gates = gates_for(service)
    failures = [k for k, v in gates.items() if result["metrics"][k] < v]
    result["gates"] = gates
    result["status"] = "failed" if failures else "passed"
    result["failed_metrics"] = failures
    if not quiet:
        mm = result["metrics"]
        print(f"[{tag}] {result['embedding']:26s} "
              f"R@1={mm['recall_at_1']:.3f} R@5={mm['recall_at_5']:.3f} MRR={mm['mrr']:.3f} "
              f"拒答准确率={mm['refusal_accuracy']:.3f} "
              f"(应拒={_fmt(mm['refusal_accuracy_on_refuse_samples'])}/应答={_fmt(mm['answer_accuracy_on_answer_samples'])}) "
              f"-> {result['status']}")
    return result


def sweep(service: HybridRagService, cases: dict) -> list[dict]:
    """拒答阈值扫描：观察准确率随阈值的变化，便于选择工作点。"""
    confs = [(c, service.search(c["query"], top_k=5)["confidence"])
             for c in cases["refusal"]]
    grid = sorted({round(v, 3) for _, v in confs})
    rows = []
    for tau in grid:
        tp = sum(1 for c, v in confs if c["expected_refused"] and v < tau)
        n_neg = sum(1 for c in cases["refusal"] if c["expected_refused"])
        n_pos = len(confs) - n_neg
        fn = sum(1 for c, v in confs if not c["expected_refused"] and v < tau)
        acc = (tp + (n_pos - fn)) / len(confs)
        rows.append({"threshold": tau, "accuracy": round(acc, 4),
                     "wrong_refuse_rate": round(fn / max(1, n_pos), 4),
                     "miss_refuse_rate": round((n_neg - tp) / max(1, n_neg), 4)})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", choices=["v1", "v2"], default="v2")
    parser.add_argument("--real", action="store_true", help="强制使用真实 BGE embedding")
    parser.add_argument("--all", action="store_true", help="跑 2 套集 x 2 种 embedding 的对比表")
    parser.add_argument("--sweep", action="store_true", help="输出拒答阈值扫描")
    args = parser.parse_args()

    primary = None
    if args.all:
        print("=" * 96)
        print("检索消融：hash-v1（离线） vs BGE（语义）")
        print("=" * 96)
        table = {}
        for tag in ("v1", "v2"):
            for label, use_real in (("hash-v1", False), ("BGE", True)):
                r = run_one(tag, use_real)
                table[f"{tag}/{label}"] = r["metrics"]
        primary = run_one(args.cases, True, quiet=True)
        json.dump(table, open(ROOT / "reports" / "rag_eval" / "ablation.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
    else:
        use_real = True if args.real else settings.use_real_embeddings
        primary = run_one(args.cases, use_real)

    if args.sweep:
        service = build_service(True, settings.embedding_model)
        rows = sweep(service, load_cases(args.cases))
        print()
        print("拒答阈值扫描（BGE, 集=" + args.cases + "）")
        print(f"{'阈值':>8s} {'准确率':>8s} {'误拒率(该答却拒)':>16s} {'漏拒率(该拒却答)':>16s}")
        for r in rows:
            if 0.45 <= r["threshold"] <= 0.85:
                print(f"{r['threshold']:8.3f} {r['accuracy']:8.3f} {r['wrong_refuse_rate']:16.3f} {r['miss_refuse_rate']:16.3f}")
        json.dump(rows, open(ROOT / "reports" / "rag_eval" / "threshold_sweep.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

    output = ROOT / "reports" / "rag_eval" / "latest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in primary.items() if not k.endswith("_results")}
    output.write_text(json.dumps(primary, ensure_ascii=False, indent=2), encoding="utf-8")

    mm = primary["metrics"]
    summary = [
        "# RAG 评测摘要", "",
        f"- 状态: **{primary['status']}**",
        f"- 评测集: `{primary['case_set']}`",
        f"- Embedding: `{primary['embedding']}`",
        f"- 拒答阈值(cos): `{primary['refusal_threshold']}`  |  相对下限: `{primary['min_score']}`",
        "",
        "| 指标 | 数值 | 护栏 |", "|---|---:|---:|",
        f"| Recall@1 | {mm['recall_at_1']:.3f} | - |",
        f"| Recall@5 | {mm['recall_at_5']:.3f} | {primary['gates'].get('recall_at_5', '-')} |",
        f"| MRR | {mm['mrr']:.3f} | {primary['gates'].get('mrr', '-')} |",
        f"| 拒答准确率（合计） | {mm['refusal_accuracy']:.3f} | {primary['gates'].get('refusal_accuracy', '-')} |",
        f"| └ 应拒答样本命中率 | {_fmt(mm['refusal_accuracy_on_refuse_samples'])} | - |",
        f"| └ 应回答样本正确率 | {_fmt(mm['answer_accuracy_on_answer_samples'])} | - |",
    ]
    (output.parent / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    try:
        import mlflow

        mlflow.set_experiment("air-quality-rag")
        with mlflow.start_run(run_name=f"hybrid-rag-eval-{primary['case_set']}"):
            mlflow.log_params({"embedding": primary["embedding"], "case_set": primary["case_set"],
                               "refusal_threshold": settings.rag_refusal_threshold})
            mlflow.log_metrics({k: v for k, v in mm.items()})
    except ImportError:
        pass
    print(json.dumps(payload["metrics"], ensure_ascii=False))
    return 0 if primary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
