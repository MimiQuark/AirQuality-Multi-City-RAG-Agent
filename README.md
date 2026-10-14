# 多城市空气质量智能问答与预警 Agent

基于 FastAPI、LangGraph、Function Calling 与混合 RAG 的多城市空气质量分析与预警 Agent。系统在没有 LLM Key 时自动降级为可重复的规则 Agent，适合本地演示和自动化评测。

## 能力

- 10 个领域工具：实时/历史数据、24 小时预测、城市对比、季节、相关性、预警、模型训练与政策检索。
- PostgreSQL + pgvector 与 pg_trgm 混合检索，支持 BGE 中文 Embedding 和确定性离线测试模型。
- 引用来源、低置信度拒答、SQLite Trace、反馈收集和 60 条工具路由回归集。
- 110 条检索样本与 44 条拒答样本，记录 Recall、MRR 和拒答准确率。

## 运行

```bash
python -m pip install -e ".[dev]"
USE_REAL_EMBEDDINGS=0 uvicorn backend.main:app --reload
python -m backend.evaluation.eval_rag
python scripts/record_checks.py
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

当前离线确定性 Embedding 用于 CI 回归。真实 BGE 检索质量、外部 LLM 回答质量和实时数据源需要配置模型/API 后单独运行，不将离线指标冒充线上效果。
