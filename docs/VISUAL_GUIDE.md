# 项目可视化与可解释性指南

本页用 Mermaid 展示空气质量 Agent 的问题路由、工具调用、RAG 拒答、trace 和评测闭环。图形全部来自当前仓库实现，不表示未完成的外部服务已经可用。

## 图 1：项目能力思维导图

```mermaid
flowchart TD
    ROOT((空气质量 AI Agent))
    ROOT --> DATA[数据底座]
    ROOT --> MODE[双模式 Agent]
    ROOT --> TOOL[工具系统]
    ROOT --> RAG[政策 RAG]
    ROOT --> TRACE[可观测性]
    ROOT --> EVAL[评测体系]

    DATA --> D1[6 个城市]
    DATA --> D2[52,704 条小时级记录]
    DATA --> D3[WAQI 实时数据接口]

    MODE --> M1[LangGraph LLM 模式]
    MODE --> M2[规则版降级模式]

    TOOL --> T1[概览 / 实时 / 预测]
    TOOL --> T2[对比 / 季节 / 相关性]
    TOOL --> T3[告警 / 综合预警]
    TOOL --> T4[模型训练 / 政策检索]

    RAG --> R1[条款知识库]
    RAG --> R2[相似度阈值]
    RAG --> R3[引用与拒答]

    TRACE --> X1[prompt / response]
    TRACE --> X2[工具调用 / 延迟]
    TRACE --> X3[SQLite trace]

    EVAL --> E1[60 条工具选择回归集]
    EVAL --> E2[多跳与 hard case]
    EVAL --> E3[bad case 迭代]
```

## 图 2：意图路由与 Agent 执行流

```mermaid
flowchart TD
    Q[用户问题] --> KEY{是否配置 LLM_API_KEY}
    KEY -->|是| LLM[LangGraph Agent]
    KEY -->|否| RULE[规则版 Agent]
    LLM --> FC[Function Calling]
    RULE --> FC
    FC --> SINGLE{单跳还是多跳}
    SINGLE -->|单跳| T1[调用单个领域工具]
    SINGLE -->|多跳| T2[组合多个领域工具]
    T1 --> POLICY[政策 RAG]
    T2 --> POLICY
    POLICY --> TH{相似度是否达到阈值}
    TH -->|是| ANSWER[带引用回答]
    TH -->|否| REFUSE[拒答 / 纯数据回退]
    ANSWER --> TRACE[(SQLite Trace)]
    REFUSE --> TRACE
```

## 图 3：10 个工具的领域分层

```mermaid
flowchart LR
    ROOT[工具注册表]
    ROOT --> DATA[数据查询层]
    ROOT --> PREDICT[预测与模型层]
    ROOT --> ALERT[告警层]
    ROOT --> KNOW[知识检索层]

    DATA --> D1[get_overview]
    DATA --> D2[get_realtime]
    DATA --> D3[get_cities_comparison]
    DATA --> D4[get_season_analysis]
    DATA --> D5[get_correlation]

    PREDICT --> P1[get_forecast]
    PREDICT --> P2[train_models]

    ALERT --> A1[get_alerts]
    ALERT --> A2[get_comprehensive_alerts]

    KNOW --> K1[query_policy]
```

工具 Schema 在 `backend/tools/implementations.py` 中统一注册，LLM 版和规则版共享同一套调用入口。

## 图 4：政策 RAG 的可解释性决策树

```mermaid
flowchart TD
    Q[政策或标准问题] --> RETRIEVE[检索知识库]
    RETRIEVE --> SCORE[归一化相似度打分]
    SCORE --> C{score ≥ threshold}
    C -->|是| CITATION[返回条款编号与依据]
    C -->|否| REFUSE[拒答或回退到数据结论]
    CITATION --> TRACE[写入 trace]
    REFUSE --> TRACE
    TRACE --> REVIEW[人工检查 bad case]
    REVIEW --> EVAL[加入回归集]
```

这条决策树解释了为什么系统不直接把相似文本交给语言模型自由生成：政策问题必须先锁定依据，再允许生成解释。

## 图 5：评测迭代与错误归因

```mermaid
flowchart LR
    R1[第一轮 48/60] --> B1[细粒度告警路由错误]
    R1 --> B2[模型训练意图缺失]
    R1 --> B3[政策工具缺失]
    B1 --> FIX1[新增 get_alerts 专属分支]
    B2 --> FIX2[新增训练意图关键词]
    B3 --> FIX3[新增 query_policy + RAG]
    FIX1 --> R2[第二轮 59/60]
    FIX2 --> R2
    FIX3 --> R2
    R2 --> FIX4[收紧政策分支的概览追加条件]
    FIX4 --> R3[第三轮 60/60]
    B1 --> REG[12 条 bad case 永久回归]
    B2 --> REG
    B3 --> REG
```

图中的 100% 只代表规则版工具选择在 60 条回归集上的结果，不代表 LLM 全链路准确率。

## 视觉阅读顺序

1. 图 1 看项目范围和数据底座。
2. 图 2 看 LLM 与规则版如何共享工具链。
3. 图 3 看 10 个工具的领域职责。
4. 图 4 看政策 RAG 为什么允许拒答。
5. 图 5 看 80% 到 100% 的真实迭代路径和归因。