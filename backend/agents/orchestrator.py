"""
Orchestrator Agent - LangGraph 多 Agent 编排器
核心流程: 用户输入 → 意图识别 + Tool 选择 → 并行执行 Tool → LLM 综合回答
"""
import os
import json
from typing import Dict, List, Any, TypedDict, Annotated
from datetime import datetime

# LangGraph 相关
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

# 导入 Tool
from backend.tools.implementations import get_tool_descriptions, execute_tool

# ============================================================
# 1. 模型配置（支持 OpenAI / DeepSeek / Qwen / Ollama）
# ============================================================

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")


def get_llm(temperature: float = 0.1):
    """获取 LLM 实例，支持国内主流 API"""
    kwargs = {
        "model": LLM_MODEL,
        "temperature": temperature,
        "api_key": LLM_API_KEY or "sk-demo",
        # 部分 Windows Python 环境（如某些 Anaconda 构建）的 openai/httpx 解压器
        # 与运行时 zlib 不兼容（Decompressor.decompress output_buffer_limit TypeError），
        # 禁用压缩响应即可绕过；curl/requests 不受影响，仅 openai 客户端触发
        "default_headers": {"Accept-Encoding": "identity"},
    }
    if LLM_BASE_URL:
        kwargs["base_url"] = LLM_BASE_URL
    return ChatOpenAI(**kwargs)


# ============================================================
# 2. State 定义
# ============================================================

class AgentState(TypedDict):
    """Agent 状态机"""
    messages: List[Any]          # 对话历史
    city: str                    # 当前上下文城市
    tool_calls: List[Dict]       # 需要执行的 Tool 调用
    tool_results: List[Dict]     # Tool 执行结果
    final_answer: str            # 最终回答
    usage: Dict[str, int]        # 累计 token 用量（prompt/completion，取不到则为 0）


# ============================================================
# 3. 系统 Prompt
# ============================================================

SYSTEM_PROMPT_V2 = """你是【多城市空气质量监测 AI 助手】，由 Orchestrator Agent 驱动。

## 可用工具
{tool_descriptions}

## 工作流
1. 分析用户问题，判断需要调用哪些工具
2. 以 JSON 数组格式输出工具调用计划: [{"tool": "工具名", "params": {"参数": "值"}}]
3. 系统会并行执行所有工具，你将收到执行结果
4. 综合结果，用中文生成专业、简洁的回答

## 规则
- 城市名需转换为 key: 北京→beijing, 上海→shanghai, 广州→guangzhou, 深圳→shenzhen, 成都→chengdu, 西安→xian
- 如果用户未指定城市，默认使用北京(beijing)
- 如果用户问"对比""排名""哪个城市"，使用 get_cities_comparison
- 如果用户问"预测""明天""未来"，使用 get_forecast
- 如果用户问"预警""告警""综合"，使用 get_comprehensive_alerts
- 如果用户问"实时""现在""当前"，使用 get_realtime
- 如果用户问"季节""冬天""夏天"，使用 get_season_analysis
- 如果用户问"相关""关系""影响"，使用 get_correlation
- 如果用户问"训练模型"，使用 train_models（仅限开发者）
- 如果用户问"标准""政策""规定""限值""法规""预案"或健康知识，使用 query_policy
- 回答必须引用具体数字，不要编造
- 凡引用环保标准/政策条款，必须标注【依据：标题(条款id)】，标题和 id 必须来自 query_policy 返回结果
- 若 query_policy 返回 refused=true，必须明说"未检索到相关标准条款，以下仅基于监测数据"，严禁编造任何条款编号或标准号
- 输出为纯文本，不要 markdown 代码块包裹

## 当前时间
{current_time}
"""


def _accumulate_usage(state: AgentState, response) -> None:
    """
    从 LLM 响应中累计 token 用量到 state["usage"]。
    不同供应商/版本的 usage_metadata 字段可能缺失，取不到就跳过（保持 0），
    接口层会把它写成 NULL，绝不因统计失败影响主流程。
    """
    try:
        meta = getattr(response, "usage_metadata", None) or {}
        usage = state.setdefault("usage", {"prompt_tokens": 0, "completion_tokens": 0})
        usage["prompt_tokens"] = usage.get("prompt_tokens", 0) + int(meta.get("input_tokens") or 0)
        usage["completion_tokens"] = usage.get("completion_tokens", 0) + int(meta.get("output_tokens") or 0)
    except Exception:
        pass


# ============================================================
# 4. LangGraph 节点函数
# ============================================================

def orchestrator_node(state: AgentState) -> AgentState:
    """
    意图识别 + Tool 选择节点
    LLM 分析用户问题，决定调用哪些 Tool
    """
    llm = get_llm(temperature=0.1)
    messages = state["messages"]
    
    # 构建系统 Prompt
    # 注意：模板内含 JSON 示例的 {} 花括号，不能用 str.format()（会把 {"tool"} 当占位符报 KeyError）
    tool_desc = json.dumps(get_tool_descriptions(), ensure_ascii=False, indent=2)
    system_msg = (SYSTEM_PROMPT_V2
                  .replace("{tool_descriptions}", tool_desc)
                  .replace("{current_time}", datetime.now().strftime("%Y-%m-%d %H:%M")))
    
    # 构造对话历史
    chat_messages = [SystemMessage(content=system_msg)]
    for msg in messages:
        if isinstance(msg, dict):
            if msg.get("role") == "user":
                chat_messages.append(HumanMessage(content=msg["content"]))
            elif msg.get("role") == "assistant":
                chat_messages.append(AIMessage(content=msg["content"]))
        elif isinstance(msg, HumanMessage):
            chat_messages.append(msg)
        elif isinstance(msg, AIMessage):
            chat_messages.append(msg)
    
    # 要求 LLM 输出 JSON 格式的工具调用计划
    planning_prompt = """\n\n请分析用户问题，输出工具调用计划。
格式要求：严格 JSON 数组，每个元素包含 "tool" 和 "params"。
如果不需要工具，输出空数组 []。
\n示例：
用户问"北京明天PM2.5怎么样"
输出：[{"tool": "get_forecast", "params": {"city": "beijing", "pollutant": "PM25"}}]
\n用户问"六个城市空气质量对比"
输出：[{"tool": "get_cities_comparison", "params": {}}]
\n用户问"上海现在空气如何，未来会超标吗"
输出：[{"tool": "get_realtime", "params": {"city": "shanghai"}}, {"tool": "get_forecast", "params": {"city": "shanghai", "pollutant": "PM25"}}]
\n请输出："""
    
    chat_messages.append(HumanMessage(content=planning_prompt))
    
    try:
        response = llm.invoke(chat_messages)
        content = response.content.strip()
        _accumulate_usage(state, response)
        
        # 提取 JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        tool_calls = json.loads(content)
        if not isinstance(tool_calls, list):
            tool_calls = []
    except Exception as e:
        # LLM 输出格式异常，降级为直接回答
        tool_calls = []
        state["messages"].append({"role": "assistant", "content": f"我在分析问题，请稍候...（解析错误: {e}）"})
    
    state["tool_calls"] = tool_calls
    return state


def tool_executor_node(state: AgentState) -> AgentState:
    """
    并行执行所有 Tool
    """
    tool_calls = state.get("tool_calls", [])
    results = []
    
    for call in tool_calls:
        tool_name = call.get("tool")
        params = call.get("params", {})

        # 默认填充城市参数——仅限声明了 city 参数的工具
        # （query_policy 等不接受 city，盲目填充会导致"不支持 city 参数"执行失败）
        if "city" not in params and state.get("city"):
            try:
                schema = next((t for t in get_tool_descriptions() if t["name"] == tool_name), None)
                if schema and "city" in (schema.get("parameters", {}).get("properties", {}) or {}):
                    params["city"] = state["city"]
            except Exception:
                pass  # schema 获取失败时不填充，不影响主流程

        result = execute_tool(tool_name, params)
        results.append({
            "tool": tool_name,
            "params": params,
            "result": result,
        })
    
    state["tool_results"] = results
    return state


def synthesizer_node(state: AgentState) -> AgentState:
    """
    综合 Tool 结果，生成最终回答
    """
    llm = get_llm(temperature=0.3)
    messages = state["messages"]
    tool_results = state.get("tool_results", [])
    
    # 构建上下文
    tool_outputs = []
    for r in tool_results:
        tool_outputs.append(f"【{r['tool']}】执行结果:\n{json.dumps(r['result'], ensure_ascii=False, indent=2)}")
    
    tool_context = "\n\n".join(tool_outputs) if tool_outputs else "（未调用任何工具）"
    
    # 获取用户最后一条问题
    user_question = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            user_question = msg["content"]
            break
        elif isinstance(msg, HumanMessage):
            user_question = msg.content
            break
    
    synthesis_prompt = f"""基于以下工具执行结果，回答用户问题。

## 用户问题
{user_question}

## 工具执行结果
{tool_context}

## 要求
- 用中文回答，专业且易懂
- 必须引用具体数字，不要编造
- 凡引用环保标准/政策条款，必须标注【依据：标题(条款id)】，标题和 id 必须来自 query_policy 返回结果
- 若 query_policy 返回 refused=true，必须明说"未检索到相关标准条款，以下仅基于监测数据"，严禁编造任何条款编号或标准号
- 如果涉及多个城市，给出对比分析
- 如果有预警/超标，明确提示风险等级和建议
- 如果工具执行失败，说明原因并提供替代信息
- 直接输出回答文本，不要加"根据数据显示"等套话
"""

    try:
        response = llm.invoke([SystemMessage(content="你是环境监测数据分析专家。"),
                               HumanMessage(content=synthesis_prompt)])
        final_answer = response.content.strip()
        _accumulate_usage(state, response)
    except Exception as e:
        final_answer = f"系统暂时无法生成完整回答，以下是原始数据:\n\n{tool_context}"
    
    state["final_answer"] = final_answer
    state["messages"].append({"role": "assistant", "content": final_answer})
    return state


def should_continue(state: AgentState) -> str:
    """
    判断是否需要继续执行 Tool
    """
    tool_calls = state.get("tool_calls", [])
    if tool_calls:
        return "tool_executor"
    return "synthesizer"


# ============================================================
# 5. 构建 LangGraph
# ============================================================

def build_agent() -> StateGraph:
    """构建并编译 Agent 工作流"""
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("tool_executor", tool_executor_node)
    workflow.add_node("synthesizer", synthesizer_node)
    
    # 设置入口
    workflow.set_entry_point("orchestrator")
    
    # 添加边
    workflow.add_conditional_edges(
        "orchestrator",
        should_continue,
        {
            "tool_executor": "tool_executor",
            "synthesizer": "synthesizer",
        }
    )
    workflow.add_edge("tool_executor", "synthesizer")
    workflow.add_edge("synthesizer", END)
    
    return workflow.compile()


# 全局 Agent 实例
_AGENT = None

def get_agent():
    """获取编译后的 Agent（单例）"""
    global _AGENT
    if _AGENT is None:
        _AGENT = build_agent()
    return _AGENT


# ============================================================
# 6. 对外接口
# ============================================================

def run_agent(question: str, city: str = "beijing", history: List[Dict] = None) -> Dict[str, Any]:
    """
    运行 Agent 处理用户问题
    
    Args:
        question: 用户问题
        city: 默认城市
        history: 历史对话记录 [{"role": "user"/"assistant", "content": "..."}]
    
    Returns:
        {"answer": str, "tool_calls": list, "tool_results": list}
    """
    agent = get_agent()
    
    messages = history or []
    messages.append({"role": "user", "content": question})
    
    state = {
        "messages": messages,
        "city": city,
        "tool_calls": [],
        "tool_results": [],
        "final_answer": "",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    }

    # 引用溯源：从工具结果中确定性地提取 references（不依赖 LLM 自觉遵守格式）
    def _build_references(tool_results: List[Dict]):
        references, refused = [], False
        for r in tool_results:
            if r.get("tool") == "query_policy":
                res = r.get("result") or {}
                if res.get("refused"):
                    refused = True
                else:
                    for d in res.get("results", []):
                        references.append({
                            "type": "standard",
                            "id": d.get("id", ""),
                            "title": d.get("title", ""),
                        })
            else:
                references.append({"type": "data", "source": r.get("tool", "")})
        return references, refused

    try:
        result = agent.invoke(state)
        tool_results = result.get("tool_results", [])
        references, refused = _build_references(tool_results)
        usage = result.get("usage") or {}
        return {
            "answer": result.get("final_answer", ""),
            "tool_calls": result.get("tool_calls", []),
            "tool_results": tool_results,
            "city": result.get("city", city),
            "references": references,
            "refused": refused,
            # token 用量透传：取不到时保持 0，由接口层写成 NULL
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens") or None,
                "completion_tokens": usage.get("completion_tokens") or None,
            },
        }
    except Exception as e:
        return {
            "answer": f"Agent 执行出错: {str(e)}",
            "tool_calls": [],
            "tool_results": [],
            "city": city,
            "references": [],
            "refused": False,
            "usage": {"prompt_tokens": None, "completion_tokens": None},
        }
