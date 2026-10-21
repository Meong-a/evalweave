# EvalWeave v1.2.2

v1.2.2 的主使用说明已整理到仓库首页 [README.md](README.md)。

本版本重点补齐：

- 公共能力为所有模型的必测项；
- 领域任务只是额外加试；
- 选择题、摘要任务可只写数据和配置；
- 全新评分规则可通过 `class_path` 插件接入，无需修改任务工厂；
- GitHub 克隆后可下载并构建完整公共数据；
- 提供医学选择题、文本摘要和自定义评分器三个可运行示例。

领域扩展操作见 [docs/ADDING_DOMAIN_TASKS_v1_2_2.md](docs/ADDING_DOMAIN_TASKS_v1_2_2.md)。
