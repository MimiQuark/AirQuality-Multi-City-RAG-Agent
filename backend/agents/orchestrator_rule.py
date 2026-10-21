"""
Rule-based Orchestrator - 无需 LLM API 的 Agent 意图识别
作为 LLM 版 Orchestrator 的降级方案，面试演示零依赖
"""
import re
from typing import Dict, List, Any
from backend.tools.implementations import execute_tool_calls_sync, is_environment_query

DOMAIN_HINTS = (
    "城市", "排名", "对比", "比较", "哪个", "季节", "相关", "告警", "预警",
    "超标", "预测", "预报", "实时", "数据来源", "模型", "算法", "原理",
    "训练", "功能", "能做什么", "你是谁", "介绍", "帮助", "help",
)


def _in_domain(text: str) -> bool:
    lower = text.lower()
    return is_environment_query(text) or any(hint in lower for hint in DOMAIN_HINTS)

# 城市名映射
CITY_MAP = {
    "北京": "beijing", "beijing": "beijing",
    "上海": "shanghai", "shanghai": "shanghai",
    "广州": "guangzhou", "guangzhou": "guangzhou",
    "深圳": "shenzhen", "shenzhen": "shenzhen",
    "成都": "chengdu", "chengdu": "chengdu",
    "西安": "xian", "xian": "xian",
}

# 污染物映射
POLLUTANT_MAP = {
    "pm2.5": "PM25", "pm25": "PM25", "细颗粒物": "PM25",
    "pm10": "PM10", "可吸入": "PM10",
    "so2": "SO2", "二氧化硫": "SO2",
    "no2": "NO2", "二氧化氮": "NO2",
    "o3": "O3", "臭氧": "O3",
    "co": "CO", "一氧化碳": "CO",
}


def extract_city(text: str, default: str = "beijing") -> str:
    """从文本中提取城市"""
    for cn, key in CITY_MAP.items():
        if cn in text:
            return key
    return default


def extract_pollutant(text: str, default: str = "PM25") -> str:
    """从文本中提取污染物"""
    text_lower = text.lower()
    for alias, key in POLLUTANT_MAP.items():
        if alias in text_lower:
            return key
    return default


def rule_based_plan(text: str, default_city: str = "beijing") -> List[Dict[str, Any]]:
    """
    基于规则的意图识别 + Tool 选择
    返回工具调用计划列表
    """
    text_lower = text.lower()
    city = extract_city(text, default_city)
    pollutant = extract_pollutant(text, "PM25")

    # 0. 领域门禁：明显不属于环保/数据问答的问题直接走政策检索，
    #    由 query_policy 返回 refused，避免规则兜底用天气数据硬答。
    if not _in_domain(text):
        return [{"tool": "query_policy", "params": {"query": text}}]

    # 1. 城市对比
    cities_in_text = [alias for alias in CITY_MAP if alias in text_lower]
    if (
        any(k in text for k in ["对比", "比较", "排名", "哪个城市", "各城市", "六个城市", "6个城市"])
        or (len(cities_in_text) >= 2 and "哪个" in text)
    ):
        return [{"tool": "get_cities_comparison", "params": {}}]

    # 2. 政策/标准/健康知识查询（RAG，可与数据工具组合）
    #    低置信时 query_policy 会返回 refused=true，由综合层执行拒答，禁止编造条款
    if any(k in text for k in ["标准", "规定", "限值", "政策", "法规", "国标", "条款", "预案"]):
        calls = [{"tool": "query_policy", "params": {"query": text}}]
        # 同时问到实际数据情况时，追加数据工具
        if any(k in text for k in ["超标", "告警", "预警"]):
            calls.append({"tool": "get_comprehensive_alerts", "params": {"city": city}})
        elif any(k in text for k in CITY_MAP) and any(k in text for k in ["多少", "怎么样", "如何", "现在", "当前", "均值", "浓度"]):
            # 仅当明确提到城市时才追加概览，纯标准问题（如"PM2.5国家标准限值"）不掺数据噪声
            calls.append({"tool": "get_overview", "params": {"city": city}})
        return calls
    
    # 2. 综合预警
    if any(k in text for k in ["综合预警", "预警中心", "所有告警", "全部预警"]):
        return [{"tool": "get_comprehensive_alerts", "params": {"city": city}}]

    # 2.5 细粒度当前告警 → get_alerts（评测 bad case 迭代新增）
    #     区分依据：问"详情/列表/明细/当前/今天"的告警是查当前阈值告警（get_alerts），
    #     泛指的告警/预警走综合预警中心（get_comprehensive_alerts）；
    #     含预测类词时不拦截（如"未来会超标吗"仍走实时+预测多跳）
    if (any(k in text for k in ["告警", "警报", "超标", "预警", "提醒"])
            and any(k in text for k in ["详情", "列表", "明细", "当前", "今天"])
            and not any(k in text for k in ["预测", "未来", "明天", "后天", "将会", "预计", "预报"])):
        return [{"tool": "get_alerts", "params": {"city": city}}]

    # 2.6 模型训练（评测 bad case 迭代新增：此前无训练分支，一律落兜底概览）
    if (any(k in text for k in ["训练", "重训", "重新学习", "建模"])
            or ("更新" in text and "模型" in text)):
        return [{"tool": "train_models", "params": {"city": city}}]

    # 3. 实时数据
    if any(k in text for k in ["实时", "现在", "当前", "此刻", "今天空气"]):
        calls = [{"tool": "get_realtime", "params": {"city": city}}]
        # 如果同时问预测，加 forecast
        if any(k in text for k in ["明天", "未来", "预测", "预报"]):
            calls.append({"tool": "get_forecast", "params": {"city": city, "pollutant": pollutant}})
        return calls
    
    # 4. 预测
    if any(k in text for k in ["预测", "预报", "明天", "后天", "未来", "将会", "预计"]):
        return [{"tool": "get_forecast", "params": {"city": city, "pollutant": pollutant}}]
    
    # 5. 告警/超标
    if any(k in text for k in ["告警", "预警", "超标", "警报", "提醒"]):
        return [{"tool": "get_comprehensive_alerts", "params": {"city": city}}]
    
    # 6. 季节分析
    if any(k in text for k in ["季节", "冬天", "夏天", "春季", "夏季", "秋季", "冬季"]):
        return [{"tool": "get_season_analysis", "params": {"city": city, "pollutant": pollutant}}]
    
    # 7. 相关性
    if any(k in text for k in ["相关", "关系", "影响", "因子", "因素"]):
        return [{"tool": "get_correlation", "params": {"city": city}}]
    
    # 8. 概览 / 默认
    return [{"tool": "get_overview", "params": {"city": city}}]


def rule_based_synthesize(question: str, tool_results: List[Dict], city: str | None = None) -> str:
    """
    基于规则的结果综合（简化版，展示 Agent 工作流）
    实际生产环境应使用 LLM 综合
    """
    if not tool_results:
        return "我理解了你的问题，但需要更多信息来回答。你可以问：\n- 北京现在空气质量怎么样\n- 上海明天PM2.5会超标吗\n- 六个城市对比\n- 成都冬季臭氧季节分析"
    
    parts = []
    for r in tool_results:
        tool_name = r["tool"]
        result = r["result"]
        
        error_text = None
        if isinstance(result, dict) and "error" in result:
            error_text = result["error"]
        elif isinstance(result, list) and result and isinstance(result[0], dict) and "error" in result[0]:
            error_text = result[0]["error"]
        if error_text is not None:
            parts.append(f"【{tool_name}】查询出错: {error_text}")
            continue

        if tool_name == "query_policy":
            # RAG 政策/标准检索结果：命中必须带【依据：标题(条款id)】，被拒必须明说
            if result.get("refused"):
                parts.append(
                    "未检索到相关标准条款，以下仅基于监测数据回答。"
                    "该问题涉及的内容不在本系统环保知识库范围内，"
                    "为避免编造条款编号，此处不引用任何标准。"
                )
            else:
                lines = ["检索到相关标准/知识条款："]
                for doc in result.get("results", []):
                    lines.append(
                        f"【依据：{doc.get('title', '')}({doc.get('id', '')})】\n"
                        f"  {doc.get('content', '').strip()}"
                    )
                parts.append("\n".join(lines))

        elif tool_name == "get_overview":
            parts.append(
                f"{result.get('name', '该城市')}近一年数据:\n"
                f"  PM2.5均值: {result.get('avg_pm25', '-')} μg/m³\n"
                f"  AQI均值: {result.get('avg_aqi', '-')}\n"
                f"  优良率: {result.get('good_days', '-')}%"
            )
        
        elif tool_name == "get_realtime":
            if result.get("live"):
                pol = result.get("pollutants", {})
                parts.append(
                    f"{result.get('source', '实时数据')}:\n"
                    f"  AQI: {result.get('aqi', '-')}\n"
                    f"  PM2.5: {pol.get('PM25', '-')} | PM10: {pol.get('PM10', '-')} | "
                    f"NO2: {pol.get('NO2', '-')} | O3: {pol.get('O3', '-')} μg/m³"
                )
            else:
                parts.append("实时数据未配置，返回历史基座最新记录。")
        
        elif tool_name == "get_forecast":
            if result.get("will_exceed"):
                parts.append(
                    f"⚠️ {result.get('station', '')}未来24小时{result.get('pollutant', '')}预警:\n"
                    f"  预计{result.get('first_exceed_in_hours', '-')}小时后超标\n"
                    f"  24h峰值: {result.get('forecast_24h_max', '-')} μg/m³ (阈值{result.get('threshold', '-')})"
                )
            else:
                parts.append(
                    f"✅ {result.get('station', '')}未来24小时{result.get('pollutant', '')}无超标风险\n"
                    f"  24h均值: {result.get('forecast_24h_mean', '-')} μg/m³ (阈值{result.get('threshold', '-')})"
                )
        
        elif tool_name == "get_cities_comparison":
            lines = ["六城市空气质量对比（近一年均值）:"]
            for s in sorted(result, key=lambda x: x.get("avg_pm25", 0), reverse=True):
                lines.append(
                    f"  {s.get('name', '')}: PM2.5={s.get('avg_pm25', '-')} "
                    f"AQI={s.get('avg_aqi', '-')} 优良率={s.get('good_days', '-')}%"
                )
            parts.append("\n".join(lines))
        
        elif tool_name == "get_comprehensive_alerts":
            pol_alerts = result.get("pollution", [])
            fc_alerts = result.get("forecast", [])
            weather_alerts = result.get("weather", [])
            
            if not pol_alerts and not fc_alerts and not weather_alerts:
                parts.append(f"{result.get('city', '')}当前无预警")
            else:
                lines = [f"{result.get('city', '')}综合预警:"]
                for a in pol_alerts:
                    lines.append(f"  🟡 {a.get('metric', '')}: {a.get('value', '-')} (IAQI={a.get('iaqi', '-')})")
                for a in fc_alerts:
                    lines.append(f"  🟠 预测{a.get('metric', '')}: {a.get('value', '-')} {a.get('basis', '')}")
                for a in weather_alerts:
                    lines.append(f"  🔵 {a.get('category', '')}{a.get('level_name', '')}: {a.get('value', '-')}{a.get('unit', '')}")
                parts.append("\n".join(lines))
        
        elif tool_name == "get_alerts":
            city_name = (result[0].get("station") if result else "") or city or ""
            if not result:
                parts.append(f"{city_name}当前无阈值告警")
            else:
                lines = [f"{city_name}当前阈值告警:"]
                for alert in result:
                    lines.append(
                        f"  {alert.get('metric', '')}: {alert.get('value', '-')} "
                        f"(阈值 {alert.get('threshold', '-')}, {alert.get('level', '')})"
                    )
                parts.append("\n".join(lines))

        elif tool_name == "train_models":
            models = result.get("models", {}) if isinstance(result, dict) else {}
            if not models:
                parts.append(f"模型训练未返回结果: {result}")
            else:
                lines = [f"{city or '该城市'}模型训练完成:"]
                for target, metrics in models.items():
                    lines.append(
                        f"  {target}: MAE={metrics.get('mae', '-')}, "
                        f"RMSE={metrics.get('rmse', '-')}, R2={metrics.get('r2', '-')}"
                    )
                parts.append("\n".join(lines))

        elif tool_name == "get_season_analysis":
            prefix = f"{city or '该城市'}季节分析" if city else "季节分析"
            lines = [f"{prefix}:"]
            for s in result:
                lines.append(f"  {s.get('season', '')}: {s.get('avg_pm25', '-')} μg/m³")
            parts.append("\n".join(lines))

        elif tool_name == "get_correlation":
            # 简化展示 PM2.5 与温度、湿度的相关性
            pm25_row = None
            for row in result:
                if row.get("indicator") == "PM25":
                    pm25_row = row
                    break
            if pm25_row:
                prefix = f"{city or '该城市'}" if city else ""
                parts.append(
                    f"{prefix}PM2.5相关性:\n"
                    f"  与温度: {pm25_row.get('temperature', '-')}\n"
                    f"  与湿度: {pm25_row.get('humidity', '-')}\n"
                    f"  与风速: {pm25_row.get('wind_speed', '-')}"
                )
    
    return "\n\n".join(parts)


def run_agent_rule_based(question: str, city: str = "beijing", **kwargs) -> Dict[str, Any]:
    """
    规则版 Agent 入口（零 LLM 依赖）
    """
    # 1. 意图识别 + Tool 选择
    tool_calls = rule_based_plan(question, city)
    
    # 2. 并发执行（与 LLM 版共用同一套参数补齐 + 并发实现）
    #    注：注释原先就写着"并行"，但实现是顺序 for 循环，此处补齐为真并发。
    tool_results = execute_tool_calls_sync(tool_calls, city)
    
    # 3. 结果综合
    answer = rule_based_synthesize(question, tool_results, city)

    # 4. 引用溯源：从工具结果中确定性地提取 references
    #    （不依赖模型自觉，规则模式下引用 100% 可追溯）
    references = []
    refused = False
    for r in tool_results:
        if r["tool"] == "query_policy":
            res = r["result"]
            if isinstance(res, dict) and res.get("refused"):
                refused = True
            elif isinstance(res, dict):
                for d in res.get("results", []):
                    references.append({
                        "type": "standard",
                        "id": d.get("id", ""),
                        "title": d.get("title", ""),
                    })
        else:
            references.append({"type": "data", "source": r["tool"]})

    return {
        "answer": answer,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "city": city,
        "mode": "rule-agent",
        "references": references,
        "refused": refused,
    }
