# EvalWeave

EvalWeave 是组内统一的大模型 Benchmark。组员只需要提供模型；EvalWeave 自动使用同一套公共试卷测试、评分并生成报告。

当前正式版本：**v1.2.2 Full Suite**。

## 先用一句话理解

```text
组员提供模型
    ↓
EvalWeave 自动运行全部公共能力测试
    ↓
如有需要，再运行该成员的领域加试
    ↓
输出总成绩、分能力成绩、逐题记录和微调前后对比
```

公共能力不是让组员自由挑选的。为了保证不同模型之间可以比较，每个模型都必须运行完整公共套件。

## 每个模型必须测试什么

| 能力 | 数据集 | 正式范围 |
|---|---|---:|
| 英文知识 | MMLU test | 14,042 |
| 中文知识 | C-Eval val | 1,346 |
| 复杂推理 | BBH | 6,511 |
| 数学推理 | GSM8K test | 1,319 |
| 真实性 | TruthfulQA | 790 |
| 指令遵循 | IFEval | 541 |
| 代码能力 | HumanEval | 164 |

C-Eval dev 260 条只用于 few-shot 示例。C-Eval test 12,342 条没有公开答案，只生成官方提交文件，不伪造本地准确率。

## 组员需要输入什么

组员只需要提供一个模型配置。

本地 Transformers 模型示例：

```json
{
  "type": "transformers",
  "id": "my-model",
  "model_path": "/data/models/my-model",
  "device": "cuda:0",
  "torch_dtype": "bfloat16",
  "local_files_only": true
}
```

OpenAI-compatible API 示例：

```json
{
  "type": "openai_compatible",
  "id": "my-api-model",
  "base_url": "http://127.0.0.1:8000/v1",
  "model": "my-api-model",
  "api_key_env": "OPENAI_API_KEY"
}
```

如果有领域加试，再额外提供领域数据。例如医学选择题、农业问答或文本摘要。领域任务不能替代公共套件。

## EvalWeave 输出什么

每次运行都会生成：

```text
predictions_v1.jsonl   每一道题：模型回答、标准答案、得分、错误信息
metrics_v1.json        总分、各能力和各学科分数
metrics_v1.csv         可以用 Excel 打开的成绩表
report_v1.html         浏览器中查看的成绩报告
hashes_v1.json         数据和 Prompt 哈希，证明使用的是同一份试卷
environment_v1.json    Python 和依赖环境
resolved_config_v1.json 实际运行配置
```

同一个基础模型和微调模型分别运行后，可以输出“提升、退化、保持正确、保持错误”的逐题对比及能力变化矩阵。

## 从 GitHub 准备完整公共数据

GitHub 保存代码、配置、来源地址和数据哈希，不直接提交几百 MB 的第三方原始数据。克隆后运行：

```bash
python3 scripts/prepare_full_data_v1_2_2.py --root "$PWD"
python3 scripts/build_full_suite_v1_2.py --root "$PWD" --source-raw "$PWD/data/raw_v1_2"
python3 scripts/validate_full_suite_v1_2.py --root "$PWD"
```

这三条命令会下载官方来源、构建完整数据并逐行验证 37,315 条规范化记录。不是抽样版。

IFEval 评分器依赖使用项目隔离目录：

```bash
bash scripts/install_ifeval_deps_v1_2.sh
bash scripts/install_ifeval_nltk_data_v1_2_1.sh
```

## 为一个模型生成全部正式实验

```bash
python3 scripts/create_full_experiments_v1_2_2.py \
  --root "$PWD" \
  --model-config "$PWD/configs/models/my_model.json" \
  --output-dir "$PWD/configs/generated/my_model" \
  --runs-dir "$PWD/runs/my_model" \
  --batch-size 8
```

运行六项可直接评分的必测任务：

```bash
bash scripts/run_benchmark_v1_2_2.sh run-suite \
  --config configs/generated/my_model/my-model_mandatory_direct_suite_full_v1_2_2.json
```

HumanEval 也是必测项，但必须把生成结果放进 Docker 安全执行环境计算 pass@1。当前用户没有 Docker 权限时，工具会拒绝在宿主机执行生成代码。

## 公共表格里没有我的领域怎么办

先判断任务形式：

| 领域任务 | 怎么接入 |
|---|---|
| 选择题 | 准备 JSONL，复制 `medical_mcq_task_v1_2_2.json` |
| 数字问答 | 使用 `numeric_qa` 配置 |
| 摘要或翻译 | 复制 `summary_task_v1_2_2.json` |
| 全新评分方法 | 复制自定义插件示例，无需修改任务工厂 |

完整的中文操作说明见 [新增领域任务指南](docs/ADDING_DOMAIN_TASKS_v1_2_2.md)。

三个可以直接验证的例子：

```bash
bash scripts/run_benchmark_v1_2_2.sh validate-task \
  --config configs/task_templates_v1_2_2/medical_mcq_task_v1_2_2.json

bash scripts/run_benchmark_v1_2_2.sh validate-task \
  --config configs/task_templates_v1_2_2/summary_task_v1_2_2.json

bash scripts/run_benchmark_v1_2_2.sh validate-task \
  --config configs/task_templates_v1_2_2/custom_keyword_task_v1_2_2.json
```

## 正式比较规则

- 所有模型必须运行相同公共套件；
- 不允许只挑对模型有利的公共任务；
- 微调前后必须使用相同数据、Prompt 和评分规则；
- 只有数据哈希和 Prompt 哈希一致的结果才能直接比较；
- 领域任务单独展示，不能冒充公共能力总分；
- TruthfulQA 必须标注为 `EvalWeave generative MC1`；
- HumanEval 生成代码不得在宿主机直接执行。

## 测试

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py' -v
```

版本变化见 [CHANGELOG_v1_2_2.md](CHANGELOG_v1_2_2.md)。
