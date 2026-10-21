# 代码与数据审查报告

> 2026-10-14 对全仓库做的一次系统性审查（静态阅读 + 数据校验 + 运行时验证）。
> 审查对象：63 个受版本控制的文件、30+ 个 HTTP 端点、4 份数据集、6 个评测脚本。

## 审查方法

1. **静态阅读**：逐文件通读核心模块（rag / agents / tools / evaluation / api），
   重点核对"注释与代码是否一致""参数语义是否唯一"。
2. **数据校验**：脚本化检查 4 份数据集的字段完整性、唯一性、值域、标注一致性。
3. **运行时验证**：把 30+ 个端点全部实测一遍（含预期失败路径），跑通全部评测与测试。

---

## 一、已修复问题（7 项）

| # | 位置 | 问题 | 影响 | 修复 |
|---|---|---|---|---|
| F1 | `backend/main.py` | `get_rag_service()` 漏传 `chunk_strategy` / `refusal_threshold` | 配置里的 `RAG_CHUNK_STRATEGY` 对 `/api/rag/search` **完全失效**，API 走 `fixed` 而 Agent 走 `structure`，两条路径的块内容不同 | 补传两个参数，已实测两条路径策略一致 |
| F2 | `backend/rag/service.py` | 两处注释与代码不符 | docstring 称"置信度取全库最高分"（实为原始余弦）；过滤注释写的是"差距不超过 (1-min_score)"（实为 `cos >= min_score × top_cos`） | 按代码实际行为改写注释 |
| F3 | `backend/agents/orchestrator.py` | 规划节点解析失败时，把异常文本当**助手发言**追加进 `state["messages"]` | 污染对话历史（多轮时会被下一轮看到），并向用户泄漏接口错误细节 | 改为写入 `state["plan_error"]` + `logger.warning`，不进对话 |
| F4 | `backend/rag/retriever.py` | `retrieve()` 的 `min_score` 与 `HybridRagService.min_score` **同名不同义**（一个是 0-1 绝对分，一个是 cos 相对比值） | 默认值下过滤恒不生效；调参时极易误用 | 默认改 `None`（不过滤），并在 docstring/注释中显式标注两者量纲不同 |
| F5 | `backend/tools/implementations.py` | 拒答说明把**排序用混合分**称作"最佳相关度"，又与 `MIN_SCORE`（相对下限，默认 0.0）比较 | 用户会看到"最佳相关度 0.62 低于阈值 0.0"这类自相矛盾的文案 | 改用 `confidence`（原始余弦）与真实拒答阈值 |
| F6 | `backend/llm_chat.py` | `build_data_context()` 每次问答都 `load_data() + preprocess()` | `/api/chat` 每次请求重读 5.2 万行 CSV 并做特征工程 | 按 engine 实例缓存 |
| F7 | `backend/agents/orchestrator.py` | 未使用导入（`asyncio`、`Annotated`）；`build_agent()` 标注返回 `StateGraph` 实为 `CompiledStateGraph` | 静态检查噪声、类型标注误导 | 清理导入、修正标注 |

**修复后回归验证**：RAG 四项指标与端到端六项指标与修复前**完全一致**，
`ruff + mypy + pytest + rag-eval` 全绿，覆盖率 89.12%。

---

## 二、已知局限（未修改，需产品决策）

| # | 事项 | 说明 | 建议 |
|---|---|---|---|
| L1 | **两套并行的 LLM 集成** | `llm_chat.py`（Dify/FastGPT，供 `/api/chat`）与 `agents/orchestrator.py`（OpenAI 兼容，供 `/api/chat/agent`）完全独立：环境变量、降级策略、引用机制都不同 | 二选一并删除另一套，或在 README 明确各自定位 |
| L2 | 两个 RAG 单例 | `main._rag_service` 与 `retriever._service` 各自持有一份索引；`/api/rag/reindex` 只重建前者 | 统一为同一个工厂（例如都走 `retriever._get_service()`） |
| L3 | `main.py` 的 `if __name__ == "__main__"` 位于文件中段（第 542 行），其后还有 3 个端点 | 现在靠 `reload=True` 的子进程导入模式侥幸可用；一旦改启动方式就会漏注册端点 | 把 `__main__` 块移文件末尾 |
| L4 | 容器内跑的是弱检索路径 | Dockerfile 只装 `requirements.txt`（**不含 sentence-transformers**），docker-compose 也未设 `USE_REAL_EMBEDDINGS` | 镜像增加 `.[rag]` 并显式设 `USE_REAL_EMBEDDINGS=1`，否则容器里是哈希向量 |
| L5 | `/api/chat/agent` 是同步端点 | FastAPI 会一直占用线程池 worker 直到 LLM 返回（实测 ~11s/次），并发上限受线程池大小限制 | 改 `async def` + `await run_agent_async()` |
| L6 | 评测集交叉 | `rag_cases_v2.json` 有 21 条 query 同时出现在检索集与拒答集，即拒答集的正样本取自检索集 | 正样本独立抽取，或接受并说明 |
| L7 | 反馈闭环只有 API | `/api/feedback` 与 `/api/rag/search` 无前端入口 | 前端补最小入口，否则"反馈->回归集"链路无实际输入 |
| L8 | `engine.load_data()` 无缓存 | 由调用方各自缓存，同一模式在 3 处重复实现 | 下沉到 engine 层做一次缓存 |
| L9 | `eval_agent.py` 的 100% | 60 条集合已被迭代到全通过，是**回归护栏**而非能力度量 | 报告中已声明，保持即可 |

---

## 三、数据审查结论

| 数据集 | 规模 | 结论 |
|---|---|---|
| `knowledge.json` | 11 篇 / 4199 字 | ✅ 字段齐全、id 唯一、小节序号连续、无空内容 |
| `rag_cases.json`（v1） | 110 检索 / 44 拒答 | ✅ 期望 id 全部存在于知识库、无重复；⚠ 拒答集全为负样本（已在报告说明） |
| `rag_cases_v2.json`（v2） | 66 检索 / 48 拒答（24/24 平衡） | ✅ 期望 id 全部存在、无重复；⚠ 21 条 query 与检索集重叠（L6） |
| `air_quality_cities.csv` | 52,704 行 × 12 列 | ✅ 6 城 × 8784 小时，时间跨度 2025-08-30 ~ 2026-08-30；**零缺失值、零负值、零越界、零重复** (city, timestamp) |
| `eval_agent.EVAL_DATASET` | 60 条 | ✅ 期望工具全部存在于 `TOOLS_REGISTRY`、问题无重复 |

---

## 四、运行时验证结论

30+ 个端点实测：**成功 20 / 预期内失败 3**

* 预期失败 3 项：`/api/rag/reindex`（未配置 `ADMIN_API_TOKEN` → 404，符合设计）、
  `POST /api/feedback` 传 `rating=9`（→ 400，校验生效）、`/api/overview?city=bogus`（→ 400，校验生效）
* 其余端点（health / cities / overview / realtime / alerts ×3 / charts ×5 / predict ×3 /
  traces ×2 / report / chat ×2 / agent tools / rag search）均返回 200 且结构正常

---

## 五、一条更有力的证据（补充进 RAG 评测报告 §5.3）

审查中实测到一组很有说明力的对照：

| 查询 | 是否领域内 | 离线 hash 余弦 |
|---|---|---:|
| PM2.5 的国家标准限值是多少 | ✅ 领域内 | **0.337** |
| 建筑抗震设计规范怎么规定 | ❌ 超纲 | **0.337** |

**两个语义毫无关系的查询得到完全相同的相似度**，比"锚点区间重叠"的表述更能说明
哈希向量为什么无法承担拒答——它在该阈值附近根本没有区分能力。

---

## 六、总体结论

* **代码质量**：无致命缺陷；7 处问题全部属于"注释与实现不一致 / 参数语义重叠 / 配置未生效"，
  已在本次审查中修复并回归验证。
* **数据质量**：4 份数据集全部通过完整性与一致性校验，仅 1 处轻微设计问题（L6）。
* **最大风险不在代码本身**，而在 L1/L4 两项：**项目同时维护两套 LLM 集成，
  且部署容器实际运行的是能力最弱的检索路径**。若用于演示或求职展示，建议优先处理这两项。
