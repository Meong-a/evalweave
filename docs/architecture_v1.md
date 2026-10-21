# 架构说明 v1

## 分层

```text
Experiment / Suite Config
        |
        +-- Model Adapter
        +-- Task Plugin
        +-- Generation Config
        |
     Benchmark Runner
        |
        +-- Prediction Schema
        +-- Scoring
        +-- Hashes and Environment
        |
   Analysis and Reporting
```

## 稳定边界

- Adapter 只负责把统一请求转换成模型调用。
- Task Plugin 只负责样本转换、Prompt、解析和评分。
- Runner 不包含领域逻辑。
- Analysis 只读取标准预测，不重新调用模型。
- Report 只读取标准指标和比较结果。

## 外部引擎

OpenCompass、lm-evaluation-harness 或 EvalScope 可以作为未来的执行后端。接入时应将它们的输出转换成 `predictions_v1.jsonl`，避免上层报告依赖某个引擎的内部目录结构。

## Schema 版本

配置、预测和报告均记录 `schema_version=1.0`。如果未来发生不兼容变更，应增加新版本转换器，不覆盖历史结果。
