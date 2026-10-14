"""
Tool 定义与实现
将现有 engine.py 的能力封装为 Function Calling Tool
"""
import os
import sys
from typing import Dict, List, Any, Optional, Type
from pydantic import BaseModel, Field

# 确保能导入原 engine 模块
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from backend import engine

# ============================================================
# 1. Tool Schema 定义（Pydantic BaseModel）
# ============================================================

class GetOverviewInput(BaseModel):
    city: str = Field(default="beijing", description="城市key，如 beijing/shanghai/guangzhou/shenzhen/chengdu/xian")

class GetForecastInput(BaseModel):
    city: str = Field(default="beijing", description="城市key")
    pollutant: str = Field(default="PM25", description="污染物: PM25/PM10/O3/NO2")

class GetCorrelationInput(BaseModel):
    city: str = Field(default="beijing", description="城市key")

class GetAlertsInput(BaseModel):
    city: str = Field(default="beijing", description="城市key")

class GetSeasonAnalysisInput(BaseModel):
    city: str = Field(default="beijing", description="城市key")
    pollutant: str = Field(default="PM25", description="污染物")

class TrainModelInput(BaseModel):
    city: str = Field(default="beijing", description="城市key")

class GetComprehensiveAlertsInput(BaseModel):
    city: str = Field(default="beijing", description="城市key")

class GetRealtimeInput(BaseModel):
    city: str = Field(default="beijing", description="城市key")

class QueryPolicyInput(BaseModel):
    query: str = Field(description="要查询的环保标准/政策/健康知识问题，如'PM2.5标准限值是多少'")


# ============================================================
# 2. Tool 实现（直接调用现有 engine + 新增能力）
# ============================================================

_df_cache = None

def _get_df():
    """全局 DataFrame 缓存"""
    global _df_cache
    if _df_cache is None:
        _df_cache = engine.preprocess(engine.load_data())
    return _df_cache


def tool_get_overview(city: str = "beijing") -> Dict[str, Any]:
    """获取指定城市的空气质量概览指标"""
    try:
        df = _get_df()
        return engine.get_city_stats(df, engine.check_city(city))
    except Exception as e:
        return {"error": str(e)}


def tool_get_forecast(city: str = "beijing", pollutant: str = "PM25") -> Dict[str, Any]:
    """获取指定城市未来24小时污染物预测预警"""
    try:
        df = _get_df()
        p = pollutant.upper().replace(".", "").replace("2_5", "25")
        if p == "PM2_5":
            p = "PM25"
        if p not in engine.POLLUTANTS:
            return {"error": f"不支持 {pollutant}，可选: {engine.POLLUTANTS}"}
        live = engine.realtime.fetch_realtime(city) if hasattr(engine, 'realtime') else None
        return engine.get_forecast_alerts(df, p, city, live=live)[0]
    except Exception as e:
        return {"error": str(e)}


def tool_get_correlation(city: str = "beijing") -> List[Dict[str, Any]]:
    """获取指定城市污染物与气象因子的相关性矩阵"""
    try:
        df = _get_df()
        return engine.get_correlation(df, engine.check_city(city))
    except Exception as e:
        return [{"error": str(e)}]


def tool_get_alerts(city: str = "beijing") -> List[Dict[str, Any]]:
    """获取指定城市当前超标告警"""
    try:
        df = _get_df()
        return engine.get_alerts(df, engine.check_city(city))
    except Exception as e:
        return [{"error": str(e)}]


def tool_get_comprehensive_alerts(city: str = "beijing") -> Dict[str, Any]:
    """获取综合预警中心（污染物+气象+预报）"""
    try:
        df = _get_df()
        city = engine.check_city(city)
        # 复用 main.py 中的逻辑，简化版
        live = None
        try:
            from backend import realtime
            live = realtime.fetch_realtime(city)
        except Exception:
            pass
        
        if live:
            vals = live["pollutants"]
            basis = f"实时数据（{live['time']}，WAQI）"
        else:
            row = engine.get_city_df(df, city).iloc[-1]
            vals = {k: float(row[k]) for k in ["PM25", "PM10", "SO2", "NO2", "CO", "O3"]}
            basis = f"历史基座最新时点（{row['timestamp']}）"
        
        pollution = engine.get_pollutant_alerts_multilevel(vals, basis)
        
        forecast = []
        for p in engine.POLLUTANTS:
            f = engine.get_forecast_alerts(df, p, city, live=live)[0]
            if f["will_exceed"]:
                forecast.append({
                    "category": "空气污染·预测",
                    "metric": engine.P_LABELS[p],
                    "value": f["forecast_24h_max"],
                    "threshold": f["threshold"],
                    "level": "orange" if f["level"] == "critical" else "yellow",
                    "basis": f"未来24h预测，预计{f['first_exceed_in_hours']}h后超标",
                })
        
        from backend import weather_alerts
        weather = weather_alerts.compute_weather_alerts(city)
        official = weather_alerts.fetch_official_warnings(city) or []
        
        return {
            "city": engine.CITIES[city],
            "basis": basis,
            "pollution": pollution,
            "forecast": forecast,
            "weather": weather,
            "official": official,
        }
    except Exception as e:
        return {"error": str(e)}


def tool_get_season_analysis(city: str = "beijing", pollutant: str = "PM25") -> List[Dict[str, Any]]:
    """获取指定城市季节分析"""
    try:
        df = _get_df()
        p = pollutant.upper().replace(".", "").replace("2_5", "25")
        if p == "PM2_5":
            p = "PM25"
        return engine.get_season_analysis(df, engine.check_city(city), p)
    except Exception as e:
        return [{"error": str(e)}]


def tool_get_cities_comparison() -> List[Dict[str, Any]]:
    """获取6城市空气质量对比"""
    try:
        df = _get_df()
        return engine.get_city_overview(df)
    except Exception as e:
        return [{"error": str(e)}]


def tool_train_models(city: str = "beijing") -> Dict[str, Any]:
    """训练指定城市全部污染物预测模型"""
    try:
        df = _get_df()
        return {"status": "ok", "models": engine.train_all(df, engine.check_city(city))}
    except Exception as e:
        return {"error": str(e)}


def tool_get_realtime(city: str = "beijing") -> Dict[str, Any]:
    """获取指定城市实时空气质量（WAQI）"""
    try:
        from backend import realtime
        result = realtime.fetch_realtime(engine.check_city(city))
        if result:
            return result
        return {"error": "未配置 WAQI_TOKEN 或获取失败"}
    except Exception as e:
        return {"error": str(e)}


def tool_query_policy(query: str) -> Dict[str, Any]:
    """
    检索环保标准/政策/健康知识库（RAG）

    幻觉控制第一道防线落地处：
    - 命中：返回带 id/title/content/score 的条款列表，供回答时引用【依据：标题(条款id)】
    - 低置信：返回 {"refused": true, "reason": ...}，上层必须明说未检索到，禁止编造条款编号
    """
    try:
        from backend.rag import retriever
        detail = retriever.retrieve_detailed(query, top_k=3)
        if detail["low_confidence"]:
            return {
                "refused": True,
                "reason": (
                    f"未检索到相关标准条款（最佳相关度 {detail['best_score']:.2f} "
                    f"低于阈值 {retriever.MIN_SCORE}），为避免编造条款，本工具拒绝凭空回答。"
                ),
                "results": [],
            }
        return {
            "refused": False,
            "results": [
                {
                    "id": d.get("id", ""),
                    "title": d.get("title", ""),
                    "content": d.get("content", ""),
                    "score": d.get("score", 0),
                }
                for d in detail["results"]
            ],
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 3. Tool 注册表（供 Agent 动态发现）
# ============================================================

TOOLS_REGISTRY = {
    "get_overview": {
        "func": tool_get_overview,
        "schema": GetOverviewInput,
        "description": "获取指定城市的空气质量概览指标（PM2.5均值、AQI、优良率等）",
    },
    "get_forecast": {
        "func": tool_get_forecast,
        "schema": GetForecastInput,
        "description": "获取指定城市未来24小时指定污染物的预测预警（是否超标、预计峰值、首次超标时间）",
    },
    "get_correlation": {
        "func": tool_get_correlation,
        "schema": GetCorrelationInput,
        "description": "获取指定城市污染物与气象因子（温度、湿度、风速）的相关性矩阵",
    },
    "get_alerts": {
        "func": tool_get_alerts,
        "schema": GetAlertsInput,
        "description": "获取指定城市当前污染物超标告警",
    },
    "get_comprehensive_alerts": {
        "func": tool_get_comprehensive_alerts,
        "schema": GetComprehensiveAlertsInput,
        "description": "获取综合预警中心，包含污染物分级预警、未来24h预测预警、高温/大风/暴雨气象预警",
    },
    "get_season_analysis": {
        "func": tool_get_season_analysis,
        "schema": GetSeasonAnalysisInput,
        "description": "获取指定城市指定污染物的季节分析（春夏秋冬均值）",
    },
    "get_cities_comparison": {
        "func": tool_get_cities_comparison,
        "schema": None,
        "description": "获取6城市（北京/上海/广州/深圳/成都/西安）的空气质量对比数据",
    },
    "train_models": {
        "func": tool_train_models,
        "schema": TrainModelInput,
        "description": "为指定城市训练全部4种污染物的预测模型（PM2.5/PM10/O3/NO2）",
    },
    "get_realtime": {
        "func": tool_get_realtime,
        "schema": GetRealtimeInput,
        "description": "获取指定城市的实时空气质量数据（需配置WAQI_TOKEN，否则返回历史最新）",
    },
    "query_policy": {
        "func": tool_query_policy,
        "schema": QueryPolicyInput,
        "description": "检索环保标准/政策法规/健康防护知识库（如PM2.5标准限值、AQI分级、重污染应急预案）；低置信时会拒绝回答(refused=true)，此时不得编造条款",
    },
}


def get_tool_descriptions() -> List[Dict[str, str]]:
    """生成供 LLM 使用的 Tool 描述列表"""
    result = []
    for name, info in TOOLS_REGISTRY.items():
        desc = {"name": name, "description": info["description"]}
        if info["schema"]:
            # 提取 schema 的字段描述
            props = {}
            for field_name, field_info in info["schema"].model_fields.items():
                props[field_name] = {
                    "type": "string" if field_info.annotation == str else "number",
                    "description": field_info.description or field_name,
                }
            desc["parameters"] = {
                "type": "object",
                "properties": props,
                "required": [k for k, v in info["schema"].model_fields.items() if v.is_required()],
            }
        else:
            desc["parameters"] = {"type": "object", "properties": {}}
        result.append(desc)
    return result


def execute_tool(name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """根据名称执行 Tool"""
    if name not in TOOLS_REGISTRY:
        return {"error": f"未知工具: {name}"}
    tool_info = TOOLS_REGISTRY[name]
    func = tool_info["func"]
    try:
        return func(**params)
    except Exception as e:
        return {"error": f"工具执行失败: {str(e)}"}
