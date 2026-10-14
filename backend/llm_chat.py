"""
LLM 智能问答模块 (Dify / FastGPT)
---------------------------------
通过环境变量配置接入 Dify 或 FastGPT 的对话 API,
将实时监测数据摘要注入对话上下文, 实现基于真实数据的智能问答。
未配置 API Key 或调用失败时, 自动降级为内置规则引擎兜底。

环境变量:
    LLM_PROVIDER     dify | fastgpt
                     （不设时自动识别: 配了 FASTGPT_API_KEY 就用 fastgpt, 否则 dify）
    DIFY_API_KEY     Dify 应用的 API Key
    DIFY_BASE_URL    默认 https://api.dify.ai/v1
    FASTGPT_API_KEY  FastGPT 应用的 API Key
    FASTGPT_BASE_URL 默认 https://api.fastgpt.in/api
                     （国内新版请以密钥页面显示的根地址为准, 如 https://cloud.fastgpt.cn/api）
    FASTGPT_APP_ID   可选。账号级(全局) API Key 需传入应用 ID,
                     在 FastGPT 应用详情页 URL 中可得: /app/detail/{appId}/...
                     应用级 API Key 无需设置
"""
import os
import requests

DIFY_API_KEY = os.getenv("DIFY_API_KEY", "")
DIFY_BASE_URL = os.getenv("DIFY_BASE_URL", "https://api.dify.ai/v1")

FASTGPT_API_KEY = os.getenv("FASTGPT_API_KEY", "")
FASTGPT_BASE_URL = os.getenv("FASTGPT_BASE_URL", "https://api.fastgpt.in/api")
FASTGPT_APP_ID = os.getenv("FASTGPT_APP_ID", "")

# 未显式指定时自动识别: 配了 FastGPT Key 就用 FastGPT
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").lower()
if not LLM_PROVIDER:
    LLM_PROVIDER = "fastgpt" if FASTGPT_API_KEY else "dify"

TIMEOUT = 30

LAST_ERROR = None  # 最近一次 LLM 调用失败原因, 供 /api/health 诊断


def llm_configured() -> bool:
    """是否已配置 LLM 服务"""
    if LLM_PROVIDER == "fastgpt":
        return bool(FASTGPT_API_KEY)
    return bool(DIFY_API_KEY)


def get_last_error() -> str:
    return LAST_ERROR or ""


def build_data_context(engine, city="beijing") -> str:
    """从引擎生成实时数据快照, 作为 LLM 对话上下文"""
    df = engine.preprocess(engine.load_data())
    cdf = engine.get_city_df(df, city)
    cname = engine.CITIES.get(city, city)

    avg_pm25 = cdf["PM25"].mean()
    dist = cdf["aqi_level"].value_counts(normalize=True) * 100
    good_pct = dist.get("优", 0) + dist.get("良", 0)
    season = cdf.groupby("season")["PM25"].mean().round(1)
    cities = engine.get_city_overview(df)
    alerts = engine.get_alerts(df, city)

    lines = [
        f"当前选中城市: {cname}。数据范围: {df['timestamp'].min()} 至 {df['timestamp'].max()}, "
        f"共 {len(df):,} 条小时级记录, {df['city'].nunique()} 个城市",
        f"{cname} PM2.5 均值: {avg_pm25:.1f} μg/m³, 空气优良率: {good_pct:.1f}%",
        f"{cname}各季节 PM2.5 均值(μg/m³): "
        + ", ".join(f"{s}季 {season.get(s, 0):.1f}" for s in ["春", "夏", "秋", "冬"]),
        "各城市对比: "
        + "; ".join(f"{s['name']} PM2.5均值{s['avg_pm25']}、AQI均值{s['avg_aqi']}、优良率{s['good_days']}%"
                    for s in cities),
        f"{cname}当前超标告警 {len(alerts)} 项: "
        + ("; ".join(f"{a['metric']}={a['value']}(阈值{a['threshold']})" for a in alerts)
           if alerts else "无"),
    ]

    # 注入实时数据快照（如已配置 WAQI 实时源）
    try:
        from .realtime import fetch_realtime
        live = fetch_realtime(city)
        if live:
            pol = ", ".join(f"{k}={v}" for k, v in live["pollutants"].items())
            lines.append(f"实时数据[{live['source']}] {live['time']}: AQI={live['aqi']}, {pol}")
    except Exception:
        pass
    return "\n".join(lines)


def _chat_dify(question: str, context: str) -> str:
    """调用 Dify 对话型应用 API"""
    resp = requests.post(
        f"{DIFY_BASE_URL}/chat-messages",
        headers={"Authorization": f"Bearer {DIFY_API_KEY}"},
        json={
            "inputs": {"data_context": context},
            "query": question,
            "response_mode": "blocking",
            "user": "env-monitor-user",
        },
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["answer"]


def _chat_fastgpt(question: str, context: str) -> str:
    """调用 FastGPT 对话 API (OpenAI 兼容格式)

    应用级 API Key 直接绑定应用, 无需 appId;
    账号级(全局) API Key 需在请求体携带 appId (FASTGPT_APP_ID)。
    """
    payload = {
        "chatId": None,
        "stream": False,
        "detail": False,
        "messages": [
            {"role": "system",
             "content": "你是环境监测数据分析助手, 基于以下实时监测数据摘要回答用户问题, "
                        "回答需引用具体数字, 不知道的内容不要编造。\n\n" + context},
            {"role": "user", "content": question},
        ],
    }
    if FASTGPT_APP_ID:
        payload["appId"] = FASTGPT_APP_ID

    resp = requests.post(
        f"{FASTGPT_BASE_URL}/v1/chat/completions",
        headers={"Authorization": f"Bearer {FASTGPT_API_KEY}"},
        json=payload,
        timeout=TIMEOUT,
    )
    if resp.status_code != 200:
        # FastGPT 错误详情在响应体 message 字段, 抛出让调用方记录
        try:
            msg = resp.json().get("message", "")
        except Exception:
            msg = resp.text[:200]
        raise RuntimeError(f"HTTP {resp.status_code}: {msg}")
    return resp.json()["choices"][0]["message"]["content"]


def answer_question_smart(question: str, engine, city="beijing") -> dict:
    """
    智能问答入口: 优先调用 LLM, 失败或未配置时降级为规则引擎。
    返回 {"answer": str, "source": "dify" | "fastgpt" | "rule"}
    """
    global LAST_ERROR
    if llm_configured():
        try:
            context = build_data_context(engine, city)
            if LLM_PROVIDER == "fastgpt":
                return {"answer": _chat_fastgpt(question, context), "source": "fastgpt"}
            return {"answer": _chat_dify(question, context), "source": "dify"}
        except Exception as e:
            LAST_ERROR = f"{type(e).__name__}: {e}"
            print(f"[llm_chat] {LLM_PROVIDER} 调用失败, 已降级规则引擎: {LAST_ERROR}")

    return {"answer": engine.answer_question(question, city), "source": "rule"}
