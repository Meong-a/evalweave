# 服务器环境审计 v1

审计日期：2026-10-21  
审计方式：SSH 只读检查

## 硬件与系统

- Ubuntu Linux 5.15
- Intel Xeon Gold 6148，80 个逻辑 CPU
- 内存 251 GiB
- GPU：8 × NVIDIA GeForce RTX 4090，单卡约 24 GiB
- NVIDIA 驱动：580.178.04
- 系统 CUDA Toolkit：11.5

## Python 环境

复用环境：`/home/share/anaconda/envs/ms-swift/bin/python`

- Python 3.11.13
- PyTorch 2.8.0，PyTorch CUDA 12.8
- Transformers 4.57.0
- Datasets 3.6.0
- Accelerate 1.10.1
- PEFT 0.17.1
- TRL 0.23.1
- Pydantic 2.11.10

检查时未安装 OpenCompass、lm-evaluation-harness、EvalScope 或 vLLM。

## 磁盘

- 根分区约 911 GiB，检查时剩余约 13 GiB。
- `/data` 约 15 TiB，检查时剩余约 524 GiB。
- 新项目应只写入 `/data/lmp/group_llm_benchmark_v1`，不向根分区安装大型依赖。

## 路线结论

v1 使用轻量通用 Runner 和现有 Transformers/PEFT 环境，不安装新依赖。OpenCompass 保留为后续可选执行后端；如果启用，应创建隔离环境并单独评估磁盘、依赖和数据缓存影响。
