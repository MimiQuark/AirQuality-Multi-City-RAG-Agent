"""
Agent 评测体系
测试规则版 Agent 的工具选择准确率
"""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.agents.orchestrator_rule import rule_based_plan, rule_based_synthesize
from backend.tools.implementations import execute_tool

# ============================================================
# 评测数据集：问题 -> 期望调用的工具
# ============================================================

EVAL_DATASET = [
    # ---------- realtime: 实时数据（6 条） ----------
    {"question": "北京现在空气质量怎么样", "expected_tools": ["get_realtime"], "city": "beijing", "category": "realtime"},
    {"question": "上海实时PM2.5浓度是多少", "expected_tools": ["get_realtime"], "city": "shanghai", "category": "realtime"},
    {"question": "guangzhou 当前空气状况如何", "expected_tools": ["get_realtime"], "city": "guangzhou", "category": "realtime"},
    {"question": "深圳此刻的AQI是多少", "expected_tools": ["get_realtime"], "city": "shenzhen", "category": "realtime"},
    {"question": "成都今天空气质量好吗", "expected_tools": ["get_realtime"], "city": "chengdu", "category": "realtime"},
    {"question": "xian 现在的一氧化碳浓度是多少", "expected_tools": ["get_realtime"], "city": "xian", "category": "realtime"},
    # ---------- forecast: 未来预测（7 条，覆盖多种污染物叫法） ----------
    {"question": "上海明天PM2.5会超标吗", "expected_tools": ["get_forecast"], "city": "shanghai", "category": "forecast"},
    {"question": "北京未来24小时臭氧预测", "expected_tools": ["get_forecast"], "city": "beijing", "category": "forecast"},
    {"question": "广州后天PM10预报", "expected_tools": ["get_forecast"], "city": "guangzhou", "category": "forecast"},
    {"question": "shenzhen 明天NO2将会如何变化", "expected_tools": ["get_forecast"], "city": "shenzhen", "category": "forecast"},
    {"question": "成都预计未来二氧化硫走势", "expected_tools": ["get_forecast"], "city": "chengdu", "category": "forecast"},
    {"question": "西安明天二氧化氮会不会超标", "expected_tools": ["get_forecast"], "city": "xian", "category": "forecast"},
    {"question": "beijing 明天空气质量预报", "expected_tools": ["get_forecast"], "city": "beijing", "category": "forecast"},
    # ---------- compare: 城市对比（6 条） ----------
    {"question": "六个城市空气质量对比", "expected_tools": ["get_cities_comparison"], "city": "beijing", "category": "compare"},
    {"question": "各城市排名", "expected_tools": ["get_cities_comparison"], "city": "beijing", "category": "compare"},
    {"question": "北京和上海哪个城市空气更好", "expected_tools": ["get_cities_comparison"], "city": "beijing", "category": "compare"},
    {"question": "比较一下广州和深圳的PM2.5水平", "expected_tools": ["get_cities_comparison"], "city": "guangzhou", "category": "compare"},
    {"question": "成都和西安空气质量对比", "expected_tools": ["get_cities_comparison"], "city": "chengdu", "category": "compare"},
    {"question": "哪个城市的臭氧污染最严重", "expected_tools": ["get_cities_comparison"], "city": "beijing", "category": "compare"},
    # ---------- alerts: 综合预警（6 条） ----------
    {"question": "成都的综合预警", "expected_tools": ["get_comprehensive_alerts"], "city": "chengdu", "category": "alerts"},
    {"question": "北京预警中心有什么消息", "expected_tools": ["get_comprehensive_alerts"], "city": "beijing", "category": "alerts"},
    {"question": "广州所有告警", "expected_tools": ["get_comprehensive_alerts"], "city": "guangzhou", "category": "alerts"},
    {"question": "深圳有哪些污染物超标", "expected_tools": ["get_comprehensive_alerts"], "city": "shenzhen", "category": "alerts"},
    {"question": "西安空气污染警报", "expected_tools": ["get_comprehensive_alerts"], "city": "xian", "category": "alerts"},
    {"question": "上海空气超标提醒", "expected_tools": ["get_comprehensive_alerts"], "city": "shanghai", "category": "alerts"},
    # ---------- season: 季节分析（6 条） ----------
    {"question": "广州冬季臭氧季节分析", "expected_tools": ["get_season_analysis"], "city": "guangzhou", "category": "season"},
    {"question": "北京夏天PM2.5的季节变化", "expected_tools": ["get_season_analysis"], "city": "beijing", "category": "season"},
    {"question": "上海春季PM10季节分析", "expected_tools": ["get_season_analysis"], "city": "shanghai", "category": "season"},
    {"question": "深圳秋季臭氧浓度季节特征", "expected_tools": ["get_season_analysis"], "city": "shenzhen", "category": "season"},
    {"question": "chengdu 冬季二氧化硫季节分析", "expected_tools": ["get_season_analysis"], "city": "chengdu", "category": "season"},
    {"question": "西安四季中哪个季节NO2最高", "expected_tools": ["get_season_analysis"], "city": "xian", "category": "season"},
    # ---------- correlation: 相关性分析（6 条） ----------
    {"question": "西安污染物和气象因子的相关性", "expected_tools": ["get_correlation"], "city": "xian", "category": "correlation"},
    {"question": "北京PM2.5与温度的关系", "expected_tools": ["get_correlation"], "city": "beijing", "category": "correlation"},
    {"question": "上海湿度和臭氧之间有什么相关性", "expected_tools": ["get_correlation"], "city": "shanghai", "category": "correlation"},
    {"question": "广州风速对PM10的影响", "expected_tools": ["get_correlation"], "city": "guangzhou", "category": "correlation"},
    {"question": "成都空气污染与气象因素分析", "expected_tools": ["get_correlation"], "city": "chengdu", "category": "correlation"},
    {"question": "shenzhen 污染物与湿度相关吗", "expected_tools": ["get_correlation"], "city": "shenzhen", "category": "correlation"},
    # ---------- overview: 概览/兜底（6 条） ----------
    {"question": "北京空气质量概览", "expected_tools": ["get_overview"], "city": "beijing", "category": "overview"},
    {"question": "上海这一年空气情况怎么样", "expected_tools": ["get_overview"], "city": "shanghai", "category": "overview"},
    {"question": "广州空气怎么样", "expected_tools": ["get_overview"], "city": "guangzhou", "category": "overview"},
    {"question": "深圳年均空气质量如何", "expected_tools": ["get_overview"], "city": "shenzhen", "category": "overview"},
    {"question": "成都空气概况", "expected_tools": ["get_overview"], "city": "chengdu", "category": "overview"},
    {"question": "xian 空气总体水平介绍", "expected_tools": ["get_overview"], "city": "xian", "category": "overview"},
    # ---------- multi_hop: 多跳问题（5 条，期望触发 2 个工具） ----------
    {"question": "深圳实时空气和未来预测", "expected_tools": ["get_realtime", "get_forecast"], "city": "shenzhen", "category": "multi_hop"},
    {"question": "上海现在空气如何，未来会超标吗", "expected_tools": ["get_realtime", "get_forecast"], "city": "shanghai", "category": "multi_hop"},
    {"question": "北京现在PM2.5多少，明天会不会更高", "expected_tools": ["get_realtime", "get_forecast"], "city": "beijing", "category": "multi_hop"},
    {"question": "广州当前空气质量，明天会好吗", "expected_tools": ["get_realtime", "get_forecast"], "city": "guangzhou", "category": "multi_hop"},
    {"question": "成都此刻臭氧浓度怎么样，未来24小时会超标吗", "expected_tools": ["get_realtime", "get_forecast"], "city": "chengdu", "category": "multi_hop"},
    # ---------- hard: 易混淆/边界样本（12 条，允许答错，如实记录） ----------
    # get_alerts 在工具注册表中存在，但规则路由没有对应分支（"告警"关键词被路由到
    # get_comprehensive_alerts，"当前/今天空气"被路由到 get_realtime），预期暴露路由合并问题
    {"question": "北京当前有哪些污染物超标告警", "expected_tools": ["get_alerts"], "city": "beijing", "category": "hard"},
    {"question": "成都PM10告警详情", "expected_tools": ["get_alerts"], "city": "chengdu", "category": "hard"},
    {"question": "广州二氧化硫超标提醒列表", "expected_tools": ["get_alerts"], "city": "guangzhou", "category": "hard"},
    {"question": "深圳今天空气有哪些告警", "expected_tools": ["get_alerts"], "city": "shenzhen", "category": "hard"},
    {"question": "上海NO2超标警报明细", "expected_tools": ["get_alerts"], "city": "shanghai", "category": "hard"},
    # train_models 同理：注册表中存在，但规则无任何关键词分支，全部落入兜底 get_overview
    {"question": "帮我训练北京的空气质量模型", "expected_tools": ["train_models"], "city": "beijing", "category": "hard"},
    {"question": "给成都重新训练模型", "expected_tools": ["train_models"], "city": "chengdu", "category": "hard"},
    {"question": "西安的模型重新学习一下", "expected_tools": ["train_models"], "city": "xian", "category": "hard"},
    {"question": "更新深圳的模型参数", "expected_tools": ["train_models"], "city": "shenzhen", "category": "hard"},
    {"question": "训练广州的空气分析模型", "expected_tools": ["train_models"], "city": "guangzhou", "category": "hard"},
    # 政策/标准类问题：系统中没有政策查询工具，应走政策查询而非实时数据，预期落入兜底
    {"question": "空气质量标准是什么", "expected_tools": ["query_policy"], "city": "beijing", "category": "hard"},
    {"question": "PM2.5的国家标准限值是多少", "expected_tools": ["query_policy"], "city": "beijing", "category": "hard"},
]


def evaluate_tool_selection():
    """
    评测工具选择准确率
    """
    total = len(EVAL_DATASET)
    correct = 0
    partial = 0
    results = []
    # 按 category 分组统计
    cat_stats = {}

    for item in EVAL_DATASET:
        question = item["question"]
        expected = set(item["expected_tools"])
        city = item.get("city", "beijing")
        category = item.get("category", "uncategorized")

        # 执行规则版意图识别
        plan = rule_based_plan(question, city)
        actual = set(call["tool"] for call in plan)

        # 判断准确率
        if actual == expected:
            status = "完全正确"
            correct += 1
        elif actual & expected:
            status = "部分正确"
            partial += 1
        else:
            status = "错误"

        results.append({
            "question": question,
            "category": category,
            "expected": list(expected),
            "actual": list(actual),
            "status": status,
        })

        stat = cat_stats.setdefault(category, {"total": 0, "correct": 0, "partial": 0})
        stat["total"] += 1
        if status == "完全正确":
            stat["correct"] += 1
        elif status == "部分正确":
            stat["partial"] += 1

    accuracy = correct / total if total else 0
    partial_accuracy = (correct + partial) / total if total else 0

    by_category = {
        cat: {
            "total": s["total"],
            "correct": s["correct"],
            "partial": s["partial"],
            "accuracy": round(s["correct"] / s["total"], 3) if s["total"] else 0,
        }
        for cat, s in cat_stats.items()
    }
    
    print("=" * 60)
    print("Agent 工具选择评测结果")
    print("=" * 60)
    for r in results:
        print(f"\n问题: {r['question']}")
        print(f"  期望: {r['expected']}")
        print(f"  实际: {r['actual']}")
        print(f"  结果: {r['status']}")
    
    print("\n" + "=" * 60)
    print(f"总计: {total} 条")
    print(f"完全正确: {correct} ({accuracy:.1%})")
    print(f"部分正确: {partial} ({partial/total:.1%})")
    print(f"完全+部分正确率: {partial_accuracy:.1%}")
    print("\n按类别分组:")
    for cat, s in by_category.items():
        print(f"  {cat}: {s['correct']}/{s['total']} ({s['accuracy']:.1%})")
    print("=" * 60)

    return {
        "total": total,
        "correct": correct,
        "partial": partial,
        "accuracy": round(accuracy, 3),
        "partial_accuracy": round(partial_accuracy, 3),
        "by_category": by_category,
        "details": results,
    }


def evaluate_end_to_end():
    """
    端到端评测：工具选择 + 执行 + 综合
    """
    test_cases = [
        {"question": "北京现在空气质量", "city": "beijing"},
        {"question": "上海明天PM2.5预测", "city": "shanghai"},
        {"question": "六个城市对比", "city": "beijing"},
    ]
    
    print("\n" + "=" * 60)
    print("端到端评测")
    print("=" * 60)
    
    for case in test_cases:
        from backend.agents.orchestrator_rule import run_agent_rule_based
        result = run_agent_rule_based(case["question"], case["city"])
        
        print(f"\n问题: {case['question']}")
        print(f"调用工具: {[t['tool'] for t in result['tool_calls']]}")
        print(f"回答:\n{result['answer'][:300]}...")
        print("-" * 40)


def main():
    """运行全部评测"""
    selection_result = evaluate_tool_selection()
    evaluate_end_to_end()
    
    # 保存结果
    output_path = os.path.join(os.path.dirname(__file__), "eval_result.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selection_result, f, ensure_ascii=False, indent=2)
    print(f"\n评测结果已保存: {output_path}")


if __name__ == "__main__":
    main()
