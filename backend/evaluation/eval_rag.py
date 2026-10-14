"""Evaluate hybrid RAG retrieval and refusal behavior, persisting every result."""
from __future__ import annotations

import json
from pathlib import Path

from backend.config import settings
from backend.rag import HybridRagService, create_embedding_provider

ROOT = Path(__file__).resolve().parents[2]
CASES = json.loads((ROOT / "backend" / "evaluation" / "rag_cases.json").read_text(encoding="utf-8"))


def compute_rank(results: list[dict], expected_id: str) -> int | None:
    for rank, item in enumerate(results, start=1):
        if item["source_id"] == expected_id:
            return rank
    return None


def main() -> int:
    service = HybridRagService(
        create_embedding_provider(settings.embedding_model, settings.use_real_embeddings),
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )
    service.index(__import__("backend.rag.service", fromlist=["load_documents"]).load_documents())
    retrieval_details = []
    recall_at_1 = recall_at_5 = 0
    reciprocal_ranks: list[float] = []
    for case in CASES["retrieval"]:
        response = service.search(case["query"], top_k=5)
        rank = compute_rank(response["results"], case["expected_source_id"])
        recall_at_1 += int(rank == 1)
        recall_at_5 += int(rank is not None and rank <= 5)
        reciprocal_ranks.append(0.0 if rank is None else 1 / rank)
        retrieval_details.append({**case, "rank": rank, "results": response["results"]})
    refusal_correct = 0
    refusal_details = []
    for case in CASES["refusal"]:
        response = service.search(case["query"], top_k=5)
        correct = bool(response["refused"]) == bool(case["expected_refused"])
        refusal_correct += int(correct)
        refusal_details.append({**case, "refused": response["refused"], "correct": correct})
    result = {
        "status": "passed",
        "embedding": service.embedding.name,
        "cases": {"retrieval": len(CASES["retrieval"]), "refusal": len(CASES["refusal"])},
        "metrics": {
            "recall_at_1": recall_at_1 / len(CASES["retrieval"]),
            "recall_at_5": recall_at_5 / len(CASES["retrieval"]),
            "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
            "refusal_accuracy": refusal_correct / len(CASES["refusal"]),
        },
        "retrieval_results": retrieval_details,
        "refusal_results": refusal_details,
    }
    thresholds = {"recall_at_5": 0.90, "mrr": 0.80, "refusal_accuracy": 0.90}
    result["status"] = "passed" if all(result["metrics"][k] >= v for k, v in thresholds.items()) else "failed"
    output = ROOT / "reports" / "rag_eval" / "latest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = [
        "# RAG 评测摘要",
        "",
        f"- 状态: **{result['status']}**",
        f"- Embedding: `{result['embedding']}`",
        f"- Recall@1: {result['metrics']['recall_at_1']:.3f}",
        f"- Recall@5: {result['metrics']['recall_at_5']:.3f}",
        f"- MRR: {result['metrics']['mrr']:.3f}",
        f"- 拒答准确率: {result['metrics']['refusal_accuracy']:.3f}",
    ]
    (output.parent / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    try:
        import mlflow

        mlflow.set_experiment("air-quality-rag")
        with mlflow.start_run(run_name="hybrid-rag-eval"):
            mlflow.log_params(
                {
                    "embedding": service.embedding.name,
                    "chunk_size": service.chunk_size,
                    "chunk_overlap": service.chunk_overlap,
                    "top_k": service.top_k,
                }
            )
            mlflow.log_metrics(result["metrics"])
            mlflow.log_artifact(str(output))
    except ImportError:
        pass
    print(json.dumps(result["metrics"], ensure_ascii=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
