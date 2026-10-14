"""
Trace 落库模块 - Agent 问答链路追踪（SQLite，零外部依赖）

用途：把每一次 /api/chat/agent 的问答记录落库，供 bad case 分析、
延迟监控、工具使用频次统计。SQLite 是标准库自带，符合项目
"不新增重依赖" 的原则。

注意：所有写库/读库操作内部都做了异常隔离，
trace 失败只记日志，绝不影响主接口响应。
"""
import os
import json
import sqlite3
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("trace")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE_DB_PATH = os.path.join(BASE_DIR, "data", "trace.db")

# 建表 SQL：tokens 字段允许 NULL（规则模式没有 token 概念，LLM 模式也可能取不到）
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,                -- ISO 格式时间戳
    question TEXT NOT NULL,          -- 用户问题
    city TEXT,                       -- 上下文城市
    mode TEXT,                       -- agent / rule-agent
    tool_calls TEXT,                 -- JSON: 工具调用计划
    answer TEXT,                     -- 最终回答
    tokens_prompt INTEGER,           -- prompt token 数（取不到则 NULL）
    tokens_completion INTEGER,       -- completion token 数（取不到则 NULL）
    latency_ms REAL,                 -- 接口耗时（毫秒）
    error TEXT                       -- 错误信息（无则 NULL）
)
"""


def _get_conn() -> sqlite3.Connection:
    """获取数据库连接（每次新建，避免跨线程共享连接问题）"""
    os.makedirs(os.path.dirname(TRACE_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(TRACE_DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """初始化数据库（服务启动时调用一次）"""
    try:
        with _get_conn() as conn:
            conn.execute(_CREATE_TABLE_SQL)
        logger.info("trace 数据库就绪: %s", TRACE_DB_PATH)
    except Exception as e:
        logger.error("trace 数据库初始化失败: %s", e)


def record_trace(
    question: str,
    city: str = "",
    mode: str = "",
    tool_calls: Optional[List[Dict]] = None,
    answer: str = "",
    tokens_prompt: Optional[int] = None,
    tokens_completion: Optional[int] = None,
    latency_ms: Optional[float] = None,
    error: Optional[str] = None,
):
    """
    写入一条问答 trace。
    任何异常都只记日志不抛出——追踪不能拖垮主服务。
    """
    try:
        with _get_conn() as conn:
            conn.execute(
                """INSERT INTO traces
                   (ts, question, city, mode, tool_calls, answer,
                    tokens_prompt, tokens_completion, latency_ms, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now().isoformat(timespec="seconds"),
                    question,
                    city,
                    mode,
                    json.dumps(tool_calls or [], ensure_ascii=False),
                    answer,
                    tokens_prompt,
                    tokens_completion,
                    round(latency_ms, 1) if latency_ms is not None else None,
                    error,
                ),
            )
    except Exception as e:
        logger.error("trace 写入失败（不影响响应）: %s", e)


def get_traces(limit: int = 50) -> List[Dict[str, Any]]:
    """查询最近的 trace 记录（按时间倒序），供 bad case 分析"""
    try:
        with _get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM traces ORDER BY id DESC LIMIT ?", (int(limit),)
            ).fetchall()
        result = []
        for r in rows:
            item = dict(r)
            # tool_calls 反序列化为对象，方便前端/分析脚本直接消费
            try:
                item["tool_calls"] = json.loads(item.get("tool_calls") or "[]")
            except Exception:
                pass
            result.append(item)
        return result
    except Exception as e:
        logger.error("trace 查询失败: %s", e)
        return []


def get_stats() -> Dict[str, Any]:
    """
    统计信息：总记录数 / 平均延迟 / 按 mode 分组 / 工具使用频次 Top
    """
    stats: Dict[str, Any] = {
        "total": 0,
        "avg_latency_ms": None,
        "by_mode": [],
        "tool_usage_top": [],
    }
    try:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n, AVG(latency_ms) AS avg_lat FROM traces"
            ).fetchone()
            stats["total"] = row["n"]
            stats["avg_latency_ms"] = round(row["avg_lat"], 1) if row["avg_lat"] else None

            stats["by_mode"] = [
                dict(r)
                for r in conn.execute(
                    """SELECT mode, COUNT(*) AS count, ROUND(AVG(latency_ms), 1) AS avg_latency_ms
                       FROM traces GROUP BY mode ORDER BY count DESC"""
                ).fetchall()
            ]

            # 工具频次：tool_calls 是 JSON 字符串，这里在 Python 侧聚合
            tool_counter: Dict[str, int] = {}
            for r in conn.execute("SELECT tool_calls FROM traces").fetchall():
                try:
                    for call in json.loads(r["tool_calls"] or "[]"):
                        name = call.get("tool", "unknown")
                        tool_counter[name] = tool_counter.get(name, 0) + 1
                except Exception:
                    continue
            stats["tool_usage_top"] = [
                {"tool": k, "count": v}
                for k, v in sorted(tool_counter.items(), key=lambda x: -x[1])[:10]
            ]
    except Exception as e:
        logger.error("trace 统计失败: %s", e)
    return stats
