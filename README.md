# 多城市空气质量智能问答与预警 Agent

基于 FastAPI、LangGraph、Function Calling 与混合 RAG 的多城市空气质量分析与预警 Agent。系统在没有 LLM Key 时自动降级为可重复的规则 Agent，适合本地演示和自动化评测。

## 能力

- 10 个领域工具：实时/历史数据、24 小时预测、城市对比、季节、相关性、预警、模型训练与政策检索。
- PostgreSQL + pgvector 与 pg_trgm 混合检索，支持 BGE 中文 Embedding 和确定性离线测试模型。
- 引用来源、低置信度拒答、SQLite Trace、反馈收集和 60 条工具路由回归集。
- 两套检索评测集：v1 简单集（110 条，题面照抄标题，仅作回归护栏）与 v2 困难集（66 条口语化提问，0% 含标题关键词）。
- 平衡拒答集 48 条（24 条超纲应拒 + 24 条领域内应答），阈值经训练/留出拆分标定。

### 实测指标（v2 困难集 + BGE 语义模型）

| 指标 | 数值 |
|---|---:|
| Recall@1 | 0.818 |
| Recall@5 | 0.955 |
| MRR | 0.870 |
| 拒答准确率 | 0.958（应拒 0.958 / 应答 0.958） |

同一套集上离线 hash 仅 Recall@1 0.561 / 拒答 0.542，差距即语义检索的价值。
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

`reports/quality/` 保存测试、覆盖率和静态检查结果；`reports/rag_eval/` 保存全部样本级检索结果与摘要；启用 MLflow 后同时记录参数和指标。

## 结果边界

* **离线确定性 Embedding（hash-v1）只用于 CI 回归**，它无法区分"领域内口语化提问"与
  "领域外问题"（实测不存在可行拒答阈值），其检索与拒答指标不代表线上能力。
* 知识库当前仅 11 篇文档，检索本质是 11 路分类，绝对指标偏高，**不要外推到生产规模**。
* 拒答阈值随 embedding 的相似度尺度变化，**换模型必须重新标定**
  （`python -m backend.evaluation.eval_rag --sweep`）。
* **端到端回答质量尚未评测**：上述指标只覆盖检索与拒答，未覆盖 LLM 生成环节的事实性与幻觉率。
* 外部 LLM 回答质量与实时数据源需配置模型/API 后单独运行，不将离线指标冒充线上效果。
