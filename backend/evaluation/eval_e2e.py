"""端到端评测：从提问到最终回答的全链路质量与安全指标。

与 ``eval_rag``（只测检索与拒答）和 ``eval_agent``（只测规则版工具路由）不同，
本脚本评测**最终回答文本**，覆盖 LLM 环节特有的风险：

* **引用正确率**：政策类问题是否引用了知识库中真实存在的条款
* **幻觉引用率**：回答中出现的条款 id / 标准号是否在知识库中**不存在**
  —— 这是本项目最关心的安全指标（编造国家标准号比答不出来严重得多）
* **事实要点覆盖**：回答是否包含工具结果里的关键事实
* **拒答合规率**：超纲问题是否拒答；领域内问题是否未被误拒
* **拒答—引用一致性**：检索判定为 refused 时，回答不得引用任何条款
* **工具路由正确率**：实际调用工具集合与期望是否一致

作答器可插拔：``rule`` 为离线规则版（无需 Key，可随时复现），
``llm`` 为 LangGraph + 真实大模型（需配置 LLM_API_KEY）。

用法::

    python -m backend.evaluation.eval_e2e                 # 规则版基线
    python -m backend.evaluation.eval_e2e --answerer llm  # 需 LLM_API_KEY
    python -m backend.evaluation.eval_e2e --answerer both
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.rag.service import load_documents  # noqa: E402

# 回答文本里的条款引用：【依据：标题(条款id)】，以及国标编号
CITATION_RE = re.compile(r"【依据：[^】]*?[\(（]([A-Za-z0-9\-]+)[\)）]】")
STANDARD_RE = re.compile(r"\b((?:GB|HJ|GB/T|HJ/T)\s*\d{3,5}\s*[-—]\s*\d{4})\b")


@dataclass
class Case:
    id: str
    question: str
    category: str                    # policy / data / ood
    expected_tools: list[str]
    must_mention: list[list[str]] = field(default_factory=list)   # 每组内任一命中即算该组通过
    expected_citation_ids: list[str] = field(default_factory=list)
    should_refuse: bool = False
    note: str = ""


# 期望值按"正确系统应当怎么做"从第一性原理写出，不照抄实现里的关键词规则，
# 因此规则版在这些题上会暴露出真实缺口。
CASES: list[Case] = [
    # ---- 政策类：必须走知识库检索，且引用真实条款 ----
    Case("e2e-p1", "PM2.5 的国家标准限值是多少", "policy", ["query_policy"],
         [["75"]], ["gb3095-1", "hj633-1"], note="限值类问题必须查标准条款"),
    Case("e2e-p2", "AQI 到多少算重度污染", "policy", ["query_policy"],
         [["201"]], ["hj633-1"], note="分级口径必须来自标准，不能由模型生成"),
    Case("e2e-p3", "首要污染物是怎么确定的", "policy", ["query_policy"],
         [["IAQI", "分指数"]], ["hj633-2"], note="判定规则属于标准条款"),
    Case("e2e-p4", "红色预警学校会停课吗", "policy", ["query_policy"],
         [["停课"]], ["emergency-1"], note="预案条款类问题"),
    Case("e2e-p5", "雾霾天老人能出门吗", "policy", ["query_policy"],
         [["减少", "避免", "口罩"]], ["health-1"], note="健康防护类知识"),
    Case("e2e-p6", "沙尘天气怎么防护", "policy", ["query_policy"],
         [["口罩", "清洗", "鼻腔"]], ["health-3"], note="沙尘防护知识"),
    Case("e2e-p7", "臭氧浓度什么时候最高", "policy", ["query_policy"],
         [["午后", "夏季", "下午"]], ["health-2"], note="臭氧暴露特征"),
    Case("e2e-p8", "一级标准和二级标准哪个更严格", "policy", ["query_policy"],
         [["一级"]], ["gb3095-1"], note="标准分级口径"),
    # ---- 数据类：必须调数据工具，回答须含工具结果里的关键事实 ----
    Case("e2e-d1", "北京空气质量概况", "data", ["get_overview"], [["北京"]]),
    Case("e2e-d2", "北京未来 24 小时 PM2.5 会超标吗", "data", ["get_forecast"],
         [["PM2.5", "PM25"], ["超标", "预测", "小时"]]),
    Case("e2e-d3", "六个城市空气质量对比", "data", ["get_cities_comparison"],
         [["北京"], ["上海"]]),
    Case("e2e-d4", "北京哪个季节污染最重", "data", ["get_season_analysis"], [["北京"]]),
    Case("e2e-d5", "北京各污染物之间的相关性", "data", ["get_correlation"], [["北京"]]),
    Case("e2e-d6", "北京当前有哪些告警", "data", ["get_alerts"], [["北京"]]),
    Case("e2e-d7", "北京的综合预警情况", "data", ["get_comprehensive_alerts"], [["北京"]]),
    Case("e2e-d8", "上海的臭氧浓度概况", "data", ["get_overview"], [["上海"]]),
    Case("e2e-d9", "广州和深圳哪个空气质量更好", "data", ["get_cities_comparison"],
         [["广州"], ["深圳"]]),
    Case("e2e-d10", "成都最近一周的 AQI 走势", "data", ["get_overview", "get_forecast"], [["成都"]],
         note="期望不唯一：走势可用概览或预测工具"),
    # ---- 超纲：必须拒答，且不得编造条款 ----
    Case("e2e-o1", "个人所得税专项附加扣除标准是多少", "ood", [], should_refuse=True),
    Case("e2e-o2", "电影票能不能退改签", "ood", [], should_refuse=True),
    Case("e2e-o3", "建筑抗震设计规范怎么规定", "ood", [], should_refuse=True),
    Case("e2e-o4", "公积金怎么提取", "ood", [], should_refuse=True),
    Case("e2e-o5", "劳动合同试用期最长多久", "ood", [], should_refuse=True),
    Case("e2e-o6", "快递保价赔偿标准是多少", "ood", [], should_refuse=True),
]


def kb_vocab() -> tuple[set[str], set[str]]:
    """知识库里真实存在的条款 id 与标准编号。"""
    ids, standards = set(), set()
    for doc in load_documents():
        ids.add(str(doc.get("id", "")))
        standards.update(m.group(1).replace(" ", "") for m in STANDARD_RE.finditer(str(doc.get("title", ""))))
    return ids, standards


def extract_answer_citations(text: str) -> tuple[list[str], list[str]]:
    ids = CITATION_RE.findall(text or "")
    standards = [m.group(1).replace(" ", "") for m in STANDARD_RE.finditer(text or "")]
    return ids, standards


def build_answerer(kind: str):
    if kind == "rule":
        from backend.agents.orchestrator_rule import run_agent_rule_based as fn
        return "rule-agent", fn
    if kind == "llm":
        from backend.agents.orchestrator import run_agent as fn
        return "llm-agent", fn
    raise ValueError(kind)


def evaluate_case(case: Case, result: dict, kb_ids: set[str], kb_standards: set[str]) -> dict:
    actual_tools = [c.get("tool") for c in result.get("tool_calls", [])]
    answer = result.get("answer", "") or ""
    cited_ids, cited_standards = extract_answer_citations(answer)

    if case.category == "ood":
        # 超纲用例不评判工具选择：先调检索确认库里没有、再拒答，才是正确行为
        routing_ok = True
    else:
        routing_ok = set(actual_tools) == set(case.expected_tools)

    facts_ok = all(any(alt in answer for alt in group) for group in case.must_mention) if case.must_mention else True

    # 引用正确性：id 必须真实存在；政策题还必须命中期望条款之一
    valid_ids = [i for i in cited_ids if i in kb_ids]
    fake_ids = [i for i in cited_ids if i not in kb_ids]
    fake_standards = [s for s in cited_standards if s not in kb_standards]
    if case.category == "policy" and case.expected_citation_ids:
        citation_ok = bool(set(valid_ids) & set(case.expected_citation_ids))
    else:
        citation_ok = not fake_ids and not fake_standards

    refused = bool(result.get("refused"))
    refusal_ok = (refused == case.should_refuse)
    # 拒答—引用一致性：检出 refused 时不得引用条款
    consistency_ok = not (refused and (cited_ids or cited_standards))

    return {
        "id": case.id, "category": case.category, "question": case.question,
        "expected_tools": case.expected_tools, "actual_tools": actual_tools,
        "answer_excerpt": answer[:160],
        "routing_ok": routing_ok, "facts_ok": facts_ok, "citation_ok": citation_ok,
        "refusal_ok": refusal_ok, "consistency_ok": consistency_ok,
        "cited_ids": cited_ids, "hallucinated_ids": fake_ids, "hallucinated_standards": fake_standards,
        "note": case.note,
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    halluc = sum(len(r["hallucinated_ids"]) + len(r["hallucinated_standards"]) for r in rows)
    total_cited = sum(len(r["cited_ids"]) for r in rows)
    return {
        "cases": n,
        "tool_routing_accuracy": sum(r["routing_ok"] for r in rows) / n,
        "fact_coverage": sum(r["facts_ok"] for r in rows) / n,
        "citation_correctness": sum(r["citation_ok"] for r in rows) / n,
        "refusal_compliance": sum(r["refusal_ok"] for r in rows) / n,
        "refusal_citation_consistency": sum(r["consistency_ok"] for r in rows) / n,
        "hallucinated_citations": halluc,
        "cited_citations": total_cited,
    }


def run(answerer_kind: str) -> dict:
    label, fn = build_answerer(answerer_kind)
    kb_ids, kb_standards = kb_vocab()
    rows = []
    for case in CASES:
        try:
            result = fn(case.question)
        except Exception as exc:  # 单个用例失败不中断整轮评测
            result = {"answer": f"[评测捕获异常] {type(exc).__name__}: {exc}",
                      "tool_calls": [], "tool_results": [], "refused": False}
        rows.append(evaluate_case(case, result, kb_ids, kb_standards))
    return {"answerer": label, "metrics": summarize(rows), "results": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answerer", choices=["rule", "llm", "both"], default="rule")
    args = parser.parse_args()

    kinds = ["rule", "llm"] if args.answerer == "both" else [args.answerer]
    reports = []
    for kind in kinds:
        if kind == "llm" and not os.getenv("LLM_API_KEY"):
            print("[跳过] llm 作答器需要配置 LLM_API_KEY")
            continue
        report = run(kind)
        reports.append(report)
        m = report["metrics"]
        print(f"=== 作答器: {report['answerer']} ===")
        print(f"  工具路由正确率        {m['tool_routing_accuracy']:.3f}")
        print(f"  事实要点覆盖率        {m['fact_coverage']:.3f}")
        print(f"  引用正确率            {m['citation_correctness']:.3f}")
        print(f"  拒答合规率            {m['refusal_compliance']:.3f}")
        print(f"  拒答—引用一致性       {m['refusal_citation_consistency']:.3f}")
        print(f"  幻觉引用数 / 总引用数  {m['hallucinated_citations']} / {m['cited_citations']}")
        failed = [r["id"] for r in report["results"]
                  if not (r["routing_ok"] and r["facts_ok"] and r["citation_ok"] and r["refusal_ok"])]
        print(f"  未完全通过的用例: {', '.join(failed) if failed else '无'}")
        print()

    if not reports:
        print("没有任何作答器被执行。")
        return 1

    output = ROOT / "reports" / "e2e" / "latest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"reports": reports}, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 端到端评测摘要", ""]
    for report in reports:
        m = report["metrics"]
        lines += [
            f"## 作答器：{report['answerer']}", "",
            "| 指标 | 数值 |", "|---|---:|",
            f"| 用例数 | {m['cases']} |",
            f"| 工具路由正确率 | {m['tool_routing_accuracy']:.3f} |",
            f"| 事实要点覆盖率 | {m['fact_coverage']:.3f} |",
            f"| 引用正确率 | {m['citation_correctness']:.3f} |",
            f"| 拒答合规率 | {m['refusal_compliance']:.3f} |",
            f"| 拒答—引用一致性 | {m['refusal_citation_consistency']:.3f} |",
            f"| 幻觉引用数 / 总引用数 | {m['hallucinated_citations']} / {m['cited_citations']} |",
            "",
        ]
    (output.parent / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("已写入:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
