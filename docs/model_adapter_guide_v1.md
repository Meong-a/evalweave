# 模型适配器指南 v1

新增适配器时继承：

```python
from group_llm_benchmark.adapters.base import ModelAdapter

class MyAdapter(ModelAdapter):
    def load(self):
        ...

    def generate(self, requests, generation_config):
        ...

    def close(self):
        ...
```

`generate` 必须：

- 为每个请求返回一个 `GenerationResponse`；
- 保留相同的 `sample_id`；
- 将单次请求错误放入 `error`，不能伪装成正常文本；
- 不在适配器内部评分；
- 不修改任务样本。

注册新适配器时，在 `adapters/factory.py` 中增加新的 `type` 映射，并增加测试。
