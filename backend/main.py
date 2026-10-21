"""
多 Agent 协作环境监测系统 - FastAPI 入口 (v2.2)
新增: /api/chat/agent  -> LangGraph Agent 智能问答
保留: /api/*           -> 原有数据接口兼容
"""
import os
import sys
import json
import time
from contextlib import asynccontextmanager

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional

# 导入原有引擎
from backend import engine
from backend.llm_chat import answer_question_smart, llm_configured, get_last_error, LLM_PROVIDER
from backend.realtime import fetch_realtime, realtime_configured
from backend.weather_alerts import compute_weather_alerts, fetch_official_warnings, qweather_configured
from backend.engine import (
    load_data, preprocess, get_city_overview, get_city_stats, get_season_analysis,
    get_correlation, get_aqi_distribution, get_latest_data, get_alerts,
    train_all, predict_single, predict_batch, get_feature_importance,
    get_chart_pm25_monthly, get_chart_diurnal_pattern, get_chart_season_boxplot,
    get_forecast_alerts, POLLUTANTS, THRESHOLDS, CITIES, DEFAULT_CITY
)

# 导入新增 Agent（自动降级：无 LLM_API_KEY 时使用规则版）
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
if LLM_API_KEY and LLM_API_KEY != "sk-demo":
    from backend.agents.orchestrator import run_agent
else:
    from backend.agents.orchestrator_rule import run_agent_rule_based as run_agent

from backend.tools.implementations import get_tool_descriptions

# 新增: trace 落库 + 每日日报（SQLite / threading 守护线程，零新增依赖）
from backend import trace as trace_store
from backend.config import settings as app_settings
from backend.feedback import record_feedback
from backend.rag import HybridRagService, create_embedding_provider, load_documents
from backend.report import generate_daily_report, DailyReportScheduler

# ============================================================
# FastAPI App（lifespan 管理: 启动时初始化 trace 库 + 拉起日报调度线程）
# ============================================================

_report_scheduler = DailyReportScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动：trace 建库（失败只记日志）+ 启动日报调度守护线程
    trace_store.init_db()
    _report_scheduler.start()
    yield
    # 关闭：优雅退出调度线程（Event 唤醒 + join，最多等 5 秒）
    _report_scheduler.stop()


app = FastAPI(
    title="多城市空气质量监测 AI 系统",
    description="多 Agent 协作架构 | Function Calling + RAG + 时序预测",
    version="2.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(app_settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(BASE_DIR, "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_df_cache = None
_rag_service: Optional[HybridRagService] = None


def get_rag_service() -> HybridRagService:
    global _rag_service
    if _rag_service is None:
        embedding = create_embedding_provider(
            app_settings.embedding_model, app_settings.use_real_embeddings
        )
        _rag_service = HybridRagService(
            embedding,
            chunk_size=app_settings.rag_chunk_size,
            chunk_overlap=app_settings.rag_chunk_overlap,
            top_k=app_settings.rag_top_k,
            chunk_strategy=app_settings.rag_chunk_strategy,
            min_score=app_settings.rag_min_score,
            refusal_threshold=app_settings.rag_refusal_threshold,
        )
        _rag_service.index(load_documents())
    return _rag_service

def get_df():
    global _df_cache
    if _df_cache is None:
        _df_cache = preprocess(load_data())
    return _df_cache


def _check_city(city: str):
    try:
        return engine.check_city(city)
    except ValueError as e:
        raise HTTPException(400, str(e))


def _check_pollutant(pollutant: str):
    p = pollutant.upper().replace(".", "")
    if p == "PM2_5":
        p = "PM25"
    if p not in POLLUTANTS:
        raise HTTPException(400, f"暂不支持 {pollutant}，可选: {POLLUTANTS}")
    return p


# ============================================================
# 模型定义
# ============================================================

class AgentChatRequest(BaseModel):
    question: str
    city: str = DEFAULT_CITY
    history: Optional[List[dict]] = None


class AgentChatResponse(BaseModel):
    answer: str
    tool_calls: List[dict]
    tool_results: List[dict]
    city: str
    mode: str = "agent"
    # 引用溯源：standard 条款引用 / data 工具来源（v2.1 新增）
    references: Optional[List[dict]] = None
    # RAG 低置信拒答标记（v2.1 新增）
    refused: bool = False


class PredictInput(BaseModel):
    PM25: float = 0
    PM10: float = 0
    SO2: float = 0
    NO2: float = 0
    CO: float = 0
    O3: float = 0
    temperature: float = 0
    humidity: float = 0
    wind_speed: float = 0
    PM25_lag1h: float = 0
    PM25_lag24h: float = 0
    PM25_roll6h: float = 0
    PM10_lag1h: float = 0
    PM10_lag24h: float = 0
    PM10_roll6h: float = 0
    O3_lag1h: float = 0
    O3_lag24h: float = 0
    O3_roll6h: float = 0
    NO2_lag1h: float = 0
    NO2_lag24h: float = 0
    NO2_roll6h: float = 0


class RagSearchRequest(BaseModel):
    query: str
    top_k: int = 5


class FeedbackRequest(BaseModel):
    trace_id: str
    rating: int
    comment: str = ""
    expected_tools: Optional[List[str]] = None


class ChatRequest(BaseModel):
    question: str
    city: str = DEFAULT_CITY


# ============================================================
# 根路由
# ============================================================

@app.get("/")
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# ============================================================
# 新增: Agent 智能问答接口 (核心亮点)
# ============================================================

@app.post("/api/chat/agent", response_model=AgentChatResponse)
def chat_agent(req: AgentChatRequest):
    """LangGraph Agent 智能问答入口（统一埋点：计时 + trace 落库）"""
    _check_city(req.city)
    t0 = time.perf_counter()
    result = None
    err = None
    try:
        result = run_agent(
            question=req.question,
            city=req.city,
            history=req.history or []
        )
        return AgentChatResponse(
            answer=result["answer"],
            tool_calls=result["tool_calls"],
            tool_results=result["tool_results"],
            city=result["city"],
            mode=result.get("mode", "agent"),
            references=result.get("references"),
            refused=result.get("refused", False),
        )
    except Exception as e:
        err = str(e)
        raise
    finally:
        # trace 落库：写库失败只记日志，绝不影响响应（record_trace 内部已隔离异常）
        latency_ms = (time.perf_counter() - t0) * 1000
        usage = (result or {}).get("usage") or {}
        trace_store.record_trace(
            question=req.question,
            city=req.city,
            mode=(result or {}).get("mode", "agent"),
            tool_calls=(result or {}).get("tool_calls", []),
            answer=(result or {}).get("answer", ""),
            tokens_prompt=usage.get("prompt_tokens"),       # LLM 模式透传，规则模式/取不到为 None
            tokens_completion=usage.get("completion_tokens"),
            latency_ms=latency_ms,
            error=err,
        )


@app.get("/api/agent/tools")
def list_agent_tools():
    """获取 Agent 可用工具列表"""
    return {
        "tools": get_tool_descriptions(),
        "cities": [{"key": k, "name": v} for k, v in CITIES.items()],
        "pollutants": POLLUTANTS,
    }


# ============================================================
# 新增: Trace 查询接口（bad case 分析）
# ============================================================

@app.get("/api/traces")
def api_traces(limit: int = 50):
    """最近的问答 trace 记录（按时间倒序）"""
    return {"traces": trace_store.get_traces(limit)}


@app.get("/api/traces/stats")
def api_trace_stats():
    """trace 统计：总数 / 平均延迟 / 按 mode 分组 / 工具使用频次 Top"""
    return trace_store.get_stats()


# ============================================================
# 新增: 每日日报接口
# ============================================================

@app.get("/api/report/daily")
def api_report_daily(date: Optional[str] = None):
    """读取日报（YYYY-MM-DD）；当天文件不存在时现场生成"""
    try:
        return generate_daily_report(date)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/report/daily/generate")
def api_report_daily_generate(
    date: Optional[str] = None,
    x_admin_token: Optional[str] = Header(None),
):
    """强制重新生成日报（覆盖当天已有文件）"""
    if app_settings.admin_api_token and x_admin_token != app_settings.admin_api_token:
        raise HTTPException(403, "admin token required")
    try:
        return generate_daily_report(date, force=True)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


# ============================================================
# 保留: 原有接口兼容
# ============================================================

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "多城市空气质量监测AI系统 v2.2",
        "chat_mode": LLM_PROVIDER if llm_configured() else "rule-fallback",
        "agent_mode": "llm" if LLM_API_KEY and LLM_API_KEY != "sk-demo" else "rule",
        "chat_error": get_last_error() if llm_configured() else "",
        "realtime": "waqi" if realtime_configured() else "offline",
        "agent": "enabled",
        "cities": [{"key": k, "name": v} for k, v in CITIES.items()],
    }


@app.get("/api/overview")
def overview(city: str = DEFAULT_CITY):
    df = get_df()
    stats = get_city_stats(df, _check_city(city))
    stats["total_records"] = len(df)
    stats["cities"] = df["city"].nunique()
    return stats


@app.get("/api/cities")
def cities():
    return get_city_overview(get_df())


@app.get("/api/cities/latest")
def latest():
    return get_latest_data(get_df())


@app.get("/api/cities/realtime")
def cities_realtime():
    if realtime_configured():
        rows = []
        for key, name in CITIES.items():
            live = fetch_realtime(key)
            if not live:
                continue
            pol = live["pollutants"]
            rows.append({
                "city": key, "station": name, "time": live["time"],
                "AQI": live["aqi"],
                "PM25": pol.get("PM25"), "PM10": pol.get("PM10"),
                "SO2": pol.get("SO2"), "NO2": pol.get("NO2"),
                "CO": pol.get("CO"), "O3": pol.get("O3"),
                "temperature": live["weather"].get("temperature"),
                "humidity": live["weather"].get("humidity"),
                "wind_speed": live["weather"].get("wind_speed"),
                "aqi_level": engine.aqi_level_name(live["aqi"]) if isinstance(live["aqi"], (int, float)) else "-",
            })
        if rows:
            return {"live": True, "source": "WAQI 实时（地面监测站）", "rows": rows}

    rows = get_latest_data(get_df())
    return {
        "live": False,
        "source": f"历史基座最新时点（CAMS 再分析，截至 {rows[0]['timestamp'][:10]}）",
        "rows": rows,
    }


@app.get("/api/season-analysis")
def season_analysis(city: str = DEFAULT_CITY, pollutant: str = "PM25"):
    return get_season_analysis(get_df(), _check_city(city), _check_pollutant(pollutant))


@app.get("/api/correlation")
def correlation(city: str = DEFAULT_CITY):
    return get_correlation(get_df(), _check_city(city))


@app.get("/api/aqi-distribution")
def aqi_distribution(city: str = DEFAULT_CITY):
    return get_aqi_distribution(get_df(), _check_city(city))


@app.get("/api/alerts")
def alerts(city: str = DEFAULT_CITY):
    city = _check_city(city)
    live = fetch_realtime(city)
    if live:
        out = []
        for key, threshold in THRESHOLDS.items():
            v = live["pollutants"].get(key)
            if isinstance(v, (int, float)) and v > threshold:
                out.append({
                    "station": CITIES[city], "metric": key, "value": v,
                    "threshold": threshold,
                    "level": "warning" if v < threshold * 1.5 else "critical",
                    "time": live["time"], "source": "live",
                })
        return out
    return get_alerts(get_df(), city)


@app.get("/api/charts/pm25-monthly")
def chart_pm25_monthly(city: str = None, pollutant: str = "PM25"):
    return get_chart_pm25_monthly(get_df(), _check_city(city) if city else None, _check_pollutant(pollutant))


@app.get("/api/charts/diurnal-pattern")
def chart_diurnal(city: str = DEFAULT_CITY, pollutant: str = "PM25"):
    return get_chart_diurnal_pattern(get_df(), _check_city(city), _check_pollutant(pollutant))


@app.get("/api/charts/season-boxplot")
def chart_season(city: str = DEFAULT_CITY, pollutant: str = "PM25"):
    return get_chart_season_boxplot(get_df(), _check_city(city), _check_pollutant(pollutant))


@app.get("/api/charts/pollutant-means")
def chart_pollutant_means(city: str = DEFAULT_CITY):
    return engine.get_chart_pollutant_means(get_df(), _check_city(city))


@app.get("/api/charts/primary-pollutant")
def chart_primary_pollutant(city: str = DEFAULT_CITY):
    return engine.get_chart_primary_pollutant(get_df(), _check_city(city))


@app.post("/api/model/train")
def api_train(city: str = DEFAULT_CITY):
    return {"status": "ok", "models": train_all(get_df(), _check_city(city))}


@app.get("/api/model/feature-importance")
def feature_importance(pollutant: str = "PM25", city: str = DEFAULT_CITY):
    p = _check_pollutant(pollutant)
    items = get_feature_importance(get_df(), p, _check_city(city))
    total = sum(i["importance"] for i in items)
    for i in items:
        i["percentage"] = round(i["importance"] / total * 100, 1)
    return items


@app.post("/api/predict")
def api_predict(data: PredictInput, pollutant: str = "PM25", city: str = DEFAULT_CITY):
    return predict_single(data.model_dump(), _check_pollutant(pollutant), _check_city(city))


@app.get("/api/predict/prefill")
def api_predict_prefill(pollutant: str = "PM25", city: str = DEFAULT_CITY):
    return engine.build_prefill(get_df(), _check_pollutant(pollutant), _check_city(city))


@app.post("/api/predict/batch")
def api_predict_batch(data: List[PredictInput], pollutant: str = "PM25", city: str = DEFAULT_CITY):
    return predict_batch([d.model_dump() for d in data], _check_pollutant(pollutant), _check_city(city))


@app.get("/api/alerts/forecast")
def alerts_forecast(pollutant: str = "PM25", city: str = DEFAULT_CITY):
    city = _check_city(city)
    p = _check_pollutant(pollutant)
    live = fetch_realtime(city)
    return get_forecast_alerts(get_df(), p, city, live=live)


@app.get("/api/alerts/comprehensive")
def alerts_comprehensive(city: str = DEFAULT_CITY):
    city = _check_city(city)
    df = get_df()

    live = fetch_realtime(city)
    if live:
        vals = live["pollutants"]
        basis = f"实时数据（{live['time']}，WAQI）"
    else:
        row = engine.get_city_df(df, city).iloc[-1]
        vals = {k: float(row[k]) for k in ["PM25", "PM10", "SO2", "NO2", "CO", "O3"]}
        basis = f"历史基座最新时点（{row['timestamp']}）"
    pollution = engine.get_pollutant_alerts_multilevel(vals, basis)

    forecast = []
    for p in POLLUTANTS:
        f = get_forecast_alerts(df, p, city, live=live)[0]
        if f["will_exceed"]:
            forecast.append({
                "category": "空气污染·预测",
                "metric": engine.P_LABELS[p],
                "value": f["forecast_24h_max"],
                "unit": "μg/m³",
                "threshold": f["threshold"],
                "level": "orange" if f["level"] == "critical" else "yellow",
                "level_name": "橙色预警" if f["level"] == "critical" else "黄色预警",
                "basis": f"未来24h预测，预计{f['first_exceed_in_hours']}h后超标",
            })

    weather = compute_weather_alerts(city)
    official = fetch_official_warnings(city) or []

    return {
        "city": CITIES[city],
        "basis": basis,
        "pollution": pollution,
        "forecast": forecast,
        "weather": weather,
        "official": official,
        "official_enabled": qweather_configured(),
    }


@app.get("/api/realtime")
def realtime(city: str = DEFAULT_CITY):
    city = _check_city(city)
    live = fetch_realtime(city)
    if live:
        alerts_list = []
        for key, threshold in THRESHOLDS.items():
            v = live["pollutants"].get(key)
            if v and v > threshold:
                alerts_list.append({
                    "metric": key, "value": v, "threshold": threshold,
                    "level": "warning" if v < threshold * 1.5 else "critical"
                })
        live["alerts"] = alerts_list
        return live

    df = engine.get_city_df(get_df(), city)
    row = df.iloc[-1]
    hist_alerts = get_alerts(get_df(), city)
    return {
        "live": False,
        "city": city,
        "source": f"历史数据最新时点（未配置实时源 WAQI_TOKEN）",
        "time": str(row["timestamp"]),
        "stations": [{
            "station": CITIES[city],
            "PM25": round(float(row["PM25"]), 1),
            "PM10": round(float(row["PM10"]), 1),
            "SO2": round(float(row["SO2"]), 1),
            "NO2": round(float(row["NO2"]), 1),
            "O3": round(float(row["O3"]), 1),
            "AQI": int(row["AQI"]),
            "aqi_level": row["aqi_level"],
        }],
        "alerts": hist_alerts,
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    """原有问答接口"""
    result = answer_question_smart(req.question, engine, _check_city(req.city))
    return {"question": req.question, **result}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)


@app.post("/api/rag/search")
def rag_search(payload: RagSearchRequest):
    """Search the policy knowledge base with vector + keyword hybrid retrieval."""
    return get_rag_service().search(payload.query, top_k=payload.top_k)


@app.post("/api/rag/reindex")
def rag_reindex(x_admin_token: str | None = Header(default=None)):
    if not app_settings.admin_api_token:
        raise HTTPException(404, "RAG 管理接口未启用")
    if x_admin_token != app_settings.admin_api_token:
        raise HTTPException(403, "管理令牌无效")
    return get_rag_service().index(load_documents())


@app.post("/api/feedback")
def feedback(payload: FeedbackRequest):
    """Persist user feedback for failed-case mining and regression evaluation."""
    if not 1 <= payload.rating <= 5:
        raise HTTPException(400, "rating 必须在 1~5")
    return record_feedback(payload.model_dump())

