# 多城市空气质量智能问答与预警 Agent

基于 FastAPI、LangGraph、Function Calling 与混合 RAG 的多城市空气质量分析与预警 Agent。系统在没有 LLM Key 时自动降级为可重复的规则 Agent，适合本地演示和自动化评测。

## 能力

- 10 个领域工具：实时/历史数据、24 小时预测、城市对比、季节、相关性、预警、模型训练与政策检索。
- PostgreSQL + pgvector 与 pg_trgm 混合检索，支持 BGE 中文 Embedding 和确定性离线测试模型。
- 引用来源、低置信度拒答、SQLite Trace、反馈收集和 60 条工具路由回归集。
- 两套检索评测集：v1 简单集（110 条，题面照抄标题，仅作回归护栏）与 v2 困难集（66 条口语化提问，0% 含标题关键词）。
- 平衡拒答集 48 条（24 条超纲应拒 + 24 条领域内应答），阈值经训练/留出拆分标定。

### 实测指标（v2 困难集 + BGE 语义模型 + 结构分块）

| 检索指标 | 数值 |
|---|---:|
| Recall@1 | 0.773 |
| Recall@5 | 0.985 |
| MRR | 0.862 |
| 拒答准确率 | 0.896（应拒 0.958 / 应答 0.833） |

同一套集上离线 hash 仅 Recall@1 0.500 / 拒答 0.562，差距即语义检索的价值。

### 端到端评测（24 用例）：规则版 vs LLM 版

| 指标 | 规则版 | LLM 版（glm-4-flash） |
|---|---:|---:|
| 工具路由正确率 | 0.667 | **0.917** |
| 事实要点覆盖率 | 0.583 | **1.000** |
| 引用正确率（回答文本内） | **0.750** | 0.667 |
| 拒答合规率 | 0.875 | **0.917** |
| 拒答—引用一致性 | **1.000** | 0.958 |
| 幻觉引用数 / 总引用（文本） | **0 / 12** | 3 / 18 |
| **API 层结构化引用正确率** | 0.750 | **0.917** |
| **API 层结构化幻觉引用** | **0** | **0** |

**核心发现**：LLM 编排把工具路由与事实覆盖分别提升 25.0 / 41.7 个百分点，
但它在**回答文本里自己写的引用不可信**（编造库外标准号、用标准号冒充条款 id）。
而 API 层的结构化引用由 `build_references()` 从工具结果确定性提取，
幻觉为 0——**引用安全应当由代码保证，而不是靠模型遵守格式**。
改进方向见报告第 9.4 节。

所有指标均可复现，且**已知短板都写在报告里**（分块收益未达统计显著、
离线 provider 无法做语义拒答、LLM 侧为非确定性单次运行）。
整改过程与全部消融数据见 **[docs/RAG_EVAL_REPORT.md](docs/RAG_EVAL_REPORT.md)**。

## 运行

```bash
python -m pip install -e ".[dev]"
USE_REAL_EMBEDDINGS=0 uvicorn backend.main:app --reload
python -m backend.evaluation.eval_rag
python scripts/record_checks.py
```

启用真实 BGE 后跑两套集的对比表与拒答阈值扫描：

```bash
python -m backend.evaluation.eval_rag --all --sweep
```

其余评测与基准：

```bash
python -m backend.evaluation.bench_chunking        # 分块策略 A/B
python -m backend.evaluation.bench_tool_parallel   # 工具并发基准
python -m backend.evaluation.eval_e2e              # 端到端（规则版基线）
python -m backend.evaluation.eval_e2e --answerer both   # 规则版 vs LLM 版对比
```

启用真实 BGE Embedding：

```bash
python -m pip install -e ".[rag,mlops]"
USE_REAL_EMBEDDINGS=1 EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5 uvicorn backend.main:app
```

## 新增接口

- `POST /api/rag/search`：混合检索并返回引用。
- `POST /api/rag/reindex`：需配置 `ADMIN_API_TOKEN` 并通过 `X-Admin-Token` 调用。
- `POST /api/feedback`：记录 trace_id、评分、评论和期望工具。

## 结果记录

`reports/quality/` 保存测试、覆盖率和静态检查结果；`reports/rag_eval/` 保存全部样本级检索结果与摘要；
`reports/bench/` 保存分块策略与工具并发基准；`reports/e2e/` 保存端到端评测明细；启用 MLflow 后同时记录参数和指标。

## 结果边界

* **离线确定性 Embedding（hash-v1）只用于 CI 回归**，它无法区分"领域内口语化提问"与
  "领域外问题"（实测不存在可行拒答阈值），其检索与拒答指标不代表线上能力。
* 知识库当前仅 11 篇文档，检索本质是 11 路分类，绝对指标偏高，**不要外推到生产规模**。
* 拒答阈值随 embedding 的相似度尺度变化，**换模型必须重新标定**
  （`python -m backend.evaluation.eval_rag --sweep`）。
* **LLM 侧为单次运行**：大模型输出非确定性，引用类指标在不同运行间有波动，
  尚未多次取平均；且未统计 token 成本。
* **分块策略的收益未达统计显著**（配对 bootstrap 95% CI 跨零），采用结构分块的依据是
  "不切断条款"的结构性正确性，不是分数提升。
* 外部 LLM 回答质量与实时数据源需配置模型/API 后单独运行，不将离线指标冒充线上效果。
