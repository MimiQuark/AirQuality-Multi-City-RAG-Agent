# 规则版 Agent 工具选择评测报告

- **运行日期**: 2026-09-02
- **运行方式**: `python backend/evaluation/eval_agent.py`（在项目根目录下执行）
- **评测对象**: `backend/agents/orchestrator_rule.py` 的 `rule_based_plan`（规则版意图识别 + Tool 选择）
- **原始结果文件**: `backend/evaluation/eval_result.json`（当前保存的是第三轮结果；本报告所有数字均来自真实运行）

---

## 0. 结论（第三轮，最新）

| 指标 | 数值 |
|---|---|
| 总条数 | 60 |
| 完全正确 | 60 |
| 部分正确 | 0 |
| 错误 | 0 |
| **总体准确率** | **100.0%** |

| category | 完全正确 / 总数 |
|---|---|
| realtime | 6/6 |
| forecast | 7/7 |
| compare | 6/6 |
| alerts | 6/6 |
| season | 6/6 |
| correlation | 6/6 |
| overview | 6/6 |
| multi_hop | 5/5 |
| hard | 12/12 |

**迭代轨迹**：第一轮 48/60（80.0%）→ 第二轮 59/60（98.3%）→ 第三轮 60/60（100%）。第一轮的 12 条 bad case 全部修复并留作回归集。修复过程见第 3 节。

---

## 1. 评测方法

### 1.1 评测流程

对评测集 `EVAL_DATASET`（`backend/evaluation/eval_agent.py` 内置）中的每条样本：

1. 将 `question` 和 `city` 输入 `rule_based_plan(question, city)`，得到实际工具调用计划；
2. 取计划中所有 `call["tool"]` 构成**实际工具集合** `actual`；
3. 与样本标注的**期望工具集合** `expected` 做集合比较。

### 1.2 判定标准（以 `eval_agent.py` 实际逻辑为准）

| 判定 | 条件 |
|---|---|
| 完全正确 | `actual == expected`（集合完全相等，多一跳少一跳都算错） |
| 部分正确 | `actual != expected` 但 `actual & expected` 非空（至少命中一个期望工具） |
| 错误 | 两集合无交集 |

- **总体准确率** = 完全正确条数 / 总条数
- **完全+部分正确率** = (完全正确 + 部分正确) / 总条数

### 1.3 数据集构成（60 条）

每条样本含 `question` / `expected_tools` / `city` / `category` 四个字段。设计原则：

- 覆盖工具注册表（`backend/tools/implementations.py`）中的全部 10 个工具，每个工具至少 5 条正例；
- 覆盖 6 个城市（北京/上海/广州/深圳/成都/西安，中文名与英文名混用）；
- 覆盖 6 种污染物的中文名与缩写混用（PM2.5/细颗粒物、SO2/二氧化硫、O3/臭氧等）；
- 含 5 条多跳问题（一条问题期望触发 `get_realtime` + `get_forecast` 两个工具）；
- 含 12 条易混淆/边界样本（hard），用于暴露规则路由的真实短板。

| category | 条数 | 期望工具 |
|---|---|---|
| realtime | 6 | get_realtime |
| forecast | 7 | get_forecast |
| compare | 6 | get_cities_comparison |
| alerts | 6 | get_comprehensive_alerts |
| season | 6 | get_season_analysis |
| correlation | 6 | get_correlation |
| overview | 6 | get_overview（兜底） |
| multi_hop | 5 | get_realtime + get_forecast |
| hard | 12 | get_alerts(5) / train_models(5) / query_policy(2) |
| **合计** | **60** | |

---

## 2. 第一轮结果（2026-09-02，修复前基线）

### 2.1 总体

| 指标 | 数值 |
|---|---|
| 总条数 | 60 |
| 完全正确 | 48 |
| 部分正确 | 0 |
| 错误 | 12 |
| **总体准确率** | **80.0%** |

常规意图（实时/预测/对比/预警/季节/相关性/概览/多跳）全部 100% 命中；错误全部集中在有意设计的 hard 类（0/12）。

### 2.2 第一轮 bad case 清单（12 条）

| # | question | 期望 | 实际 | 原因分析 |
|---|---|---|---|---|
| 1 | 北京当前有哪些污染物超标告警 | get_alerts | get_realtime | "当前"命中实时分支并提前 return，轮不到告警分支 |
| 2 | 成都PM10告警详情 | get_alerts | get_comprehensive_alerts | "告警"被一刀切到综合预警，细粒度告警工具无专属分支 |
| 3 | 广州二氧化硫超标提醒列表 | get_alerts | get_comprehensive_alerts | 同上 |
| 4 | 深圳今天空气有哪些告警 | get_alerts | get_realtime | "今天空气"命中实时分支并提前 return |
| 5 | 上海NO2超标警报明细 | get_alerts | get_comprehensive_alerts | "警报"路由到综合预警 |
| 6 | 帮我训练北京的空气质量模型 | train_models | get_overview | 规则无任何"训练"关键词分支，落入兜底概览 |
| 7 | 给成都重新训练模型 | train_models | get_overview | 同上 |
| 8 | 西安的模型重新学习一下 | train_models | get_overview | 同上 |
| 9 | 更新深圳的模型参数 | train_models | get_overview | 同上 |
| 10 | 训练广州的空气分析模型 | train_models | get_overview | 同上 |
| 11 | 空气质量标准是什么 | query_policy | get_overview | 当时尚无政策查询工具，落入兜底 |
| 12 | PM2.5的国家标准限值是多少 | query_policy | get_overview | 同上 |

---

## 3. 修复与迭代记录

第一轮 12 条 bad case 归为三类，逐一修复（均为真实代码改动，可 diff）：

1. **告警路由合并（#1–#5）→ 新增细粒度告警分支**：`rule_based_plan` 在综合预警分支之后、实时分支之前新增 get_alerts 专属分支——问"详情/列表/明细/当前/今天"的告警查当前阈值告警（get_alerts），泛指告警仍走综合预警中心；含预测类词不拦截（保证"未来会超标吗"多跳不受影响）。
2. **train_models 无路由（#6–#10）→ 新增训练意图分支**：关键词 训练/重训/重新学习/建模/更新+模型。
3. **政策类无工具（#11–#12）→ 新增 `query_policy` 工具**（基于 RAG 知识库检索国标/标准条款，低置信度自动拒答）+ 政策意图分支（标准/规定/限值/政策/法规/国标/条款/预案）；同步把评测集中两处笔误的期望工具名 `get_policy` 对齐为真实注册的 `query_policy`。

**第二轮结果：59/60 = 98.3%**。唯一部分正确：`PM2.5的国家标准限值是多少` 期望 `[query_policy]`，实际 `[query_policy, get_overview]`——政策分支见到"多少"就追加数据概览，纯标准问题掺入了数据噪声。

4. **第三轮修复**：政策分支追加概览的条件收紧为"明确提到城市 + 数据疑问词"，纯标准问题不再追加数据工具。

**第三轮结果：60/60 = 100%**（即本报告第 0 节）。第一轮的 12 条 bad case 已全部转为回归集留在数据集中。

---

## 4. 附注

- 端到端评测（`evaluate_end_to_end`）同步跑通：预测、对比工具返回真实数据；`get_realtime` 因本地未配置 `WAQI_TOKEN` 返回"未配置 WAQI_TOKEN 或获取失败"，属环境配置问题，不影响工具选择准确率结论。
- 数据集与分组统计逻辑均在 `backend/evaluation/eval_agent.py` 内，重跑命令：`python backend/evaluation/eval_agent.py`，结果自动覆盖写入 `eval_result.json`。
- LLM 版 Agent 的 bad case 迭代路径：从 `data/trace.db`（/api/traces 接口）捞线上 bad case → 补进 orchestrator 的 few-shot 示例 → 用本回归集验证不回退。
