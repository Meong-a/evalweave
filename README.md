# EvalWeave v1

EvalWeave 是一个模型无关、领域无关的统一大模型 Benchmark 工具。它通过统一的模型适配器、任务插件、样本/预测 schema 和可追溯运行产物，让组内不同成员提交的模型结果能够横向比较，也能对同一模型微调前后的能力变化进行纵向分析。

## 核心能力

- 支持 Hugging Face Transformers、PEFT/LoRA、OpenAI-compatible API 和 Mock 模型。
- 支持多项选择与文本生成任务，并可扩展自定义任务插件和评分器。
- 支持单模型评测、多模型排行榜、多任务能力矩阵和 base/tuned 成对比较。
- 成对比较同时报告目标能力提升、非目标能力变化、改对/改错样本、Bootstrap 置信区间和 McNemar 近似检验。
- 每次运行保存逐题输出、汇总指标、解析后配置、环境快照以及配置、数据和 Prompt 哈希。
- 输出 JSON、CSV 和自包含 HTML 报告；运行目录带时间戳，不覆盖历史结果。

EvalWeave 的核心不包含任何特定模型、成员或领域逻辑。医学、法律、金融、通用知识等场景都通过配置和插件接入；“能力退化分析”是比较模块之一，不是项目的唯一用途。

## 1. 快速体验（无需模型或额外依赖）

要求 Python 3.10+。克隆项目后执行：

```bash
git clone https://github.com/Meong-a/evalweave.git
cd evalweave
export PYTHONPATH="$PWD/src"

python -m unittest discover -s tests -p "test_*.py" -v
python -m group_llm_benchmark run \
  --config configs/experiments/mock_baseline_public_demo_v1.json
python -m group_llm_benchmark run-suite \
  --config configs/suites/core_smoke_suite_v1.json
```

也可以安装为命令行工具：

```bash
python -m pip install -e .
evalweave --help
```

`gbench` 是保留的兼容别名。项目不会自动下载模型、数据，也不会启动推理服务。

## 2. 接入一个模型

复制最接近的版本化模板：

- Transformers：`configs/models/transformers_local_template_v1.json`
- PEFT/LoRA：`configs/models/peft_lora_template_v1.json`
- OpenAI-compatible：`configs/models/openai_compatible_template_v1.json`

例如本地 Transformers 模型：

```json
{
  "type": "transformers",
  "id": "my-model-v1",
  "model_path": "/absolute/path/to/model",
  "device": "cuda:0",
  "torch_dtype": "bfloat16",
  "local_files_only": true
}
```

LoRA 模型在上述配置中增加 `adapter_path`，并把 `type` 设为 `peft`。

## 3. 接入一个任务

任务数据推荐使用 JSONL，每行一个样本。例如选择题：

```json
{"id":"q1","question":"2 + 2 = ?","choices":["3","4","5","6"],"answer":"B","source":"demo"}
```

创建 `configs/tasks/my_task_v1.json`：

```json
{
  "type": "mcq",
  "name": "my_task_v1",
  "dataset_path": "../../data/my_task_v1/test_v1.jsonl",
  "id_field": "id",
  "question_field": "question",
  "choices_field": "choices",
  "answer_field": "answer",
  "source_field": "source",
  "capabilities": ["reasoning", "domain_knowledge"],
  "primary_metric": "accuracy",
  "prompt_template": "Question: {question}\nOptions:\n{choices}\nAnswer:"
}
```

文本生成任务使用 `type: "text_generation"`，可配置 `exact_match`、`normalized_exact_match`、`token_f1`、`rouge_l` 和 `bleu4`。

## 4. 运行一次评测

创建 `configs/experiments/my_model_my_task_v1.json`：

```json
{
  "schema_version": "1.0",
  "experiment_name": "my_model_my_task_v1",
  "model_ref": "../models/my_model_v1.json",
  "task_ref": "../tasks/my_task_v1.json",
  "generation": {"do_sample": false, "max_new_tokens": 64},
  "batch_size": 4,
  "seed": 42,
  "resume": true,
  "output_root": "../../runs"
}
```

先验证，再运行：

```bash
evalweave validate-config --config configs/experiments/my_model_my_task_v1.json
evalweave run --config configs/experiments/my_model_my_task_v1.json
```

运行结果位于：

```text
runs/<timestamp>_<experiment>_<config-hash>/
├── resolved_config_v1.json
├── environment_v1.json
├── hashes_v1.json
├── predictions_v1.jsonl
├── metrics_v1.json
├── metrics_v1.csv
├── report_v1.html
└── run_log_v1.jsonl
```

## 5. 多模型与多任务评测

Suite 配置包含多个实验：

```bash
evalweave run-suite --config configs/suites/core_smoke_suite_v1.json
```

- `mode: "leaderboard"`：在相同任务、Prompt 和指标下做模型横向排名。
- `mode: "matrix"`：汇总模型 × 任务 × 能力，避免对不可比指标强行排序。

## 6. 比较微调前后变化

分别运行 base 和 tuned 实验，然后执行：

```bash
evalweave compare \
  --baseline runs/<base-run> \
  --candidate runs/<tuned-run> \
  --output-root comparisons
```

比较要求两次运行具有相同的 `sample_id` 集合和主指标。输出包括总体变化、能力变化矩阵、改对/改错样本、统计置信区间以及 HTML 报告，可同时分析目标能力增益和非目标能力变化。

## 7. 可复现约定

- 数据、Prompt、生成参数、解析规则不一致时，不应直接比较结果。
- 恢复中断运行必须显式传入原运行目录；EvalWeave 会核对配置、数据和 Prompt 哈希。
- API 模型可能由服务端更新，应在模型配置元数据中记录确切服务版本。
- `report_v1.html` 用于阅读，`predictions_v1.jsonl` 与 `metrics_v1.json` 是审计依据。

更详细说明见 `docs/architecture_v1.md`、`docs/model_adapter_guide_v1.md`、`docs/task_plugin_guide_v1.md` 和 `docs/reproducibility_v1.md`。
