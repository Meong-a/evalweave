# 给 EvalWeave 增加领域任务

组员不需要选择公共能力。MMLU、C-Eval、BBH、GSM8K、TruthfulQA、IFEval 和 HumanEval 属于组内统一公共考试。

领域任务是公共考试之外的加试，例如医学问答、农业知识、法律推理或文本摘要。

## 先判断你的任务属于哪一种

| 你的数据 | 使用的任务类型 | 是否需要写 Python |
|---|---|---|
| 有问题、选项和标准答案 | `mcq_formal` | 不需要 |
| 有问题和一个数字答案 | `numeric_qa` | 不需要 |
| 有输入文本和参考输出，如摘要、翻译 | `text_generation` | 不需要 |
| 需要全新的数据格式或评分规则 | `custom` | 需要写一个插件类 |

## 选择题：只准备数据和配置

每行一道题：

```json
{"id":"medical_001","question":"问题","choices":{"A":"选项A","B":"选项B"},"answer":"A","source":"medical"}
```

复制 `configs/task_templates_v1_2_2/medical_mcq_task_v1_2_2.json`，修改任务名称、数据路径和能力名称，然后验证：

```bash
bash scripts/run_benchmark_v1_2_2.sh validate-task --config /absolute/my_medical_task.json
```

看到 `status: PASS` 和正确题数，就表示接入成功。

## 摘要或翻译：只准备输入和参考答案

```json
{"id":"summary_001","input":"原文","reference":"人工参考摘要","source":"news"}
```

复制 `configs/task_templates_v1_2_2/summary_task_v1_2_2.json`。内置可选指标为：

- `exact_match`
- `normalized_exact_match`
- `token_f1`
- `rouge_l`
- `bleu4`

摘要示例默认使用 ROUGE-L 和 Token F1。

## 全新评分规则：使用自定义插件

如果现有评分方法都不合适，复制：

```text
src/group_llm_benchmark/extensions/keyword_coverage_v1_2_2.py
configs/task_templates_v1_2_2/custom_keyword_task_v1_2_2.json
```

任务配置通过下面这一行找到插件：

```json
"class_path": "your_python_module:YourTaskClass"
```

插件继承 `TaskPlugin` 并实现四件事：读取样本、生成 Prompt、解析模型回答、计算分数。无需再修改 `factory.py`。

## 领域任务如何进入正式实验

公共套件始终完整运行。领域任务另建一个实验配置，使用同一个模型配置，再把领域结果与公共能力报告一起展示。不能用领域任务替代公共套件，也不能只挑对模型有利的公共任务。

新增任务至少要保存：

- 数据来源和授权说明；
- 数据版本及 SHA-256；
- Prompt 模板；
- 主要评分指标；
- 完整数据量；
- 一项能够加载全部样本的测试。
