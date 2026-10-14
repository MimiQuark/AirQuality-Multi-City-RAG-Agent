"""
每日空气质量日报模块

- generate_daily_report(): 复用现有工具链路（execute_tool）对 6 城市生成 Markdown 日报
- DailyReportScheduler: 每天定时自动生成日报的后台守护线程

调度选型说明（项目哲学：50 行自建优于引入重依赖）：
APScheduler 提供 cron/持久化 jobstore/分布式等能力，远超"每天固定时间
生成一次日报"这个需求。因此用 threading 守护线程自实现：每 60 秒检查
一次是否到点、当天是否已生成。零新增 pip 依赖，行为完全可控、可测。
"""
import os
import threading
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from backend.tools.implementations import execute_tool

logger = logging.getLogger("report")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(BASE_DIR, "data", "reports")

# 每日自动生成时间（07:30）：上班前生成，早间查看时日报已就绪
REPORT_TIME = "07:30"
# 调度线程检查间隔（秒）：60 秒粒度对"每天一次"的任务足够精确，且几乎零开销
CHECK_INTERVAL_SECONDS = 60

# 6 个监测城市（与 engine.CITIES 保持一致）
REPORT_CITIES = ["beijing", "shanghai", "guangzhou", "shenzhen", "chengdu", "xian"]

# 数据文件是历史基座，日报中注明数据口径，避免误读为实时
DATA_BASIS_NOTE = "数据口径：CAMS 再分析历史基座（近一年小时级），预警/预测基于机器学习模型"


def _report_path(date_str: str) -> str:
    return os.path.join(REPORT_DIR, f"{date_str}.md")


def _try_llm_summary(md_context: str) -> Optional[str]:
    """
    有 LLM_API_KEY 时生成一段综述；无 key 或调用失败时返回 None（静默降级）。
    """
    api_key = os.getenv("LLM_API_KEY", "")
    if not api_key or api_key == "sk-demo":
        return None
    try:
        from backend.agents.orchestrator import get_llm
        from langchain_core.messages import HumanMessage
        llm = get_llm(temperature=0.3)
        prompt = (
            "你是环境监测数据分析专家。基于以下 6 城市空气质量日报数据，"
            "写一段 150 字以内的中文综述：总体状况、需要关注的城市与风险、出行建议。\n\n"
            + md_context
        )
        return llm.invoke([HumanMessage(content=prompt)]).content.strip()
    except Exception as e:
        logger.warning("日报 LLM 综述生成失败（降级为纯数据日报）: %s", e)
        return None


def generate_daily_report(date: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
    """
    生成指定日期的空气质量日报（Markdown）。

    Args:
        date: YYYY-MM-DD，缺省为今天
        force: True 时即使当天文件已存在也强制重新生成

    Returns:
        {"date", "path", "content", "cached", "tools_used"}
    """
    date_str = date or datetime.now().strftime("%Y-%m-%d")
    path = _report_path(date_str)

    # 同日复用已生成文件（force 除外）
    if os.path.exists(path) and not force:
        with open(path, "r", encoding="utf-8") as f:
            return {"date": date_str, "path": path, "content": f.read(),
                    "cached": True, "tools_used": []}

    tools_used = []

    # ── 1. 六城市概览对比 ─────────────────────────────
    comparison = execute_tool("get_cities_comparison", {})
    tools_used.append("get_cities_comparison")

    # ── 2. 逐城市：综合预警 + 未来24h PM2.5 预测 ──────
    city_alerts: Dict[str, Any] = {}
    city_forecasts: Dict[str, Any] = {}
    for c in REPORT_CITIES:
        city_alerts[c] = execute_tool("get_comprehensive_alerts", {"city": c})
        city_forecasts[c] = execute_tool("get_forecast", {"city": c, "pollutant": "PM25"})
    tools_used += ["get_comprehensive_alerts", "get_forecast"]

    # ── 3. 拼装 Markdown ─────────────────────────────
    lines = [
        f"# 多城市空气质量日报（{date_str}）",
        "",
        f"> {DATA_BASIS_NOTE}",
        "",
        "## 一、六城市概览（近一年均值）",
        "",
        "| 城市 | PM2.5均值 | PM10均值 | AQI均值 | 优良率 |",
        "| --- | --- | --- | --- | --- |",
    ]
    if isinstance(comparison, list):
        for s in comparison:
            lines.append(
                f"| {s.get('name', s.get('city', '-'))} "
                f"| {s.get('avg_pm25', '-')} μg/m³ "
                f"| {s.get('avg_pm10', '-')} μg/m³ "
                f"| {s.get('avg_aqi', '-')} "
                f"| {s.get('good_days', '-')}% |"
            )
    else:
        lines.append(f"| （概览数据获取失败: {comparison}） | | | | |")

    lines += ["", "## 二、预警汇总", ""]
    any_alert = False
    for c in REPORT_CITIES:
        info = city_alerts.get(c) or {}
        if "error" in info:
            lines.append(f"- **{c}**：预警数据获取失败（{info['error']}）")
            continue
        cname = info.get("city", c)
        pol = info.get("pollution", [])
        fc = info.get("forecast", [])
        if not pol and not fc:
            lines.append(f"- **{cname}**：当前无预警 ✅")
        else:
            any_alert = True
            for a in pol:
                lines.append(
                    f"- **{cname}** 🟡 {a.get('metric', '')}: {a.get('value', '-')}"
                    f"{a.get('unit', '')}（IAQI={a.get('iaqi', '-')}，{a.get('level_name', '')}）"
                )
            for a in fc:
                lines.append(
                    f"- **{cname}** 🟠 预测 {a.get('metric', '')} 超标风险: "
                    f"峰值 {a.get('value', '-')} μg/m³（{a.get('basis', '')}）"
                )
    if not any_alert:
        lines.append("")
        lines.append("六城市当前均无污染物超标预警。")

    lines += ["", "## 三、未来 24 小时超标风险提示（PM2.5）", ""]
    for c in REPORT_CITIES:
        f = city_forecasts.get(c) or {}
        if "error" in f:
            lines.append(f"- **{c}**：预测失败（{f['error']}）")
        elif f.get("will_exceed"):
            lines.append(
                f"- ⚠️ **{f.get('station', c)}**：预计 {f.get('first_exceed_in_hours', '-')} 小时后超标，"
                f"24h 峰值 {f.get('forecast_24h_max', '-')} μg/m³（阈值 {f.get('threshold', '-')}）"
            )
        else:
            lines.append(
                f"- ✅ **{f.get('station', c)}**：无超标风险，"
                f"24h 均值 {f.get('forecast_24h_mean', '-')} μg/m³（阈值 {f.get('threshold', '-')}）"
            )

    # ── 4. LLM 综述（可选，无 key 时自动跳过）─────────
    summary = _try_llm_summary("\n".join(lines))
    if summary:
        lines += ["", "## 四、AI 综述", "", summary]

    lines += [
        "",
        "---",
        f"数据来源工具：{', '.join(sorted(set(tools_used)))}",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]

    content = "\n".join(lines)

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(content)
    logger.info("日报已生成: %s", path)

    return {"date": date_str, "path": path, "content": content,
            "cached": False, "tools_used": sorted(set(tools_used))}


class DailyReportScheduler:
    """
    每日日报后台调度线程（threading 守护线程自实现，替代 APScheduler）。

    工作方式：每 60 秒醒来一次，若当前时间已过 REPORT_TIME 且当天日报
    尚未生成，则生成当天日报。服务启动时若当天已到点但文件缺失，会立即
    补生成（catch-up），保证"当天必有一份日报"。
    线程内所有异常都被捕获，绝不影响 web 服务。
    """

    def __init__(self):
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _due_today(self, now: datetime) -> bool:
        """今天是否已到生成时间"""
        hh, mm = REPORT_TIME.split(":")
        return (now.hour, now.minute) >= (int(hh), int(mm))

    def _loop(self):
        while not self._stop_event.is_set():
            try:
                now = datetime.now()
                today = now.strftime("%Y-%m-%d")
                # 到点且当天文件不存在才生成（文件存在 = 已生成，跳过）
                if self._due_today(now) and not os.path.exists(_report_path(today)):
                    generate_daily_report(today)
            except Exception as e:
                # 调度线程的任何异常都不能外溢影响 web 服务
                logger.error("日报调度循环异常（已忽略，下轮重试）: %s", e)
            # 用 Event.wait 代替 sleep：停止信号可以立刻唤醒线程，实现优雅退出
            self._stop_event.wait(CHECK_INTERVAL_SECONDS)

    def start(self):
        """启动守护线程（幂等）"""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="daily-report-scheduler", daemon=True
        )
        self._thread.start()
        logger.info("日报调度线程已启动（每天 %s 生成，每 %ds 检查一次）",
                    REPORT_TIME, CHECK_INTERVAL_SECONDS)

    def stop(self):
        """优雅退出：发停止信号并等待线程结束"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("日报调度线程已停止")
