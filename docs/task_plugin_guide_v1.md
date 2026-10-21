# 任务插件指南 v1

任务插件继承 `TaskPlugin` 并实现：

- `load_samples`：转换成统一 `Sample`；
- `build_request`：构造 Prompt 或消息；
- `parse_response`：将原始输出转为可评分答案；
- `score`：返回一个或多个数值指标。

每个样本必须提供稳定且唯一的 `sample_id`。能力标签应描述可跨任务比较的能力，例如 `reasoning`、`summarization`、`coding`、`domain_knowledge` 和 `instruction_following`。

不要在核心 Runner 中增加领域判断。领域逻辑应留在插件中。
