### 任务流程
1. 在[GQA数据集](https://cs.stanford.edu/people/dorarad/gqa/download.html)基础上，增加思维链
2. SFT使得[Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)接收图片和问题作为输入，输出思维链和最终的答案
3. 针对思维链模式改进GRPO，促进模型推理能力的涌现

### 7.26任务梳理
1. 数据集部分
   * [ ] （周）调用Qwen2.5-VL-72B-Instruct的API，通过设计的提示词模板，获取针对问题的思维链，最终的数据集应该包括`image_id`图片编号、`cot`问题的思维链、`answer`针对问题的答案。同时需要准备好训练集和测试集。
   * [ ] （周）设计思维链评估指标和流程。
   * [ ] （周）将一部分数据集上传到服务器，注意文件组织，从数据集中的`image_id`找到图片的绝对路径写到数据集中。
   * [ ] （脱）写数据集预处理代码，可以参考`reinforced_cot/utils/preprocess.py`，重点包括：加上合适的提示词（You are an expert in visual language ...）、按照Qwen2.5-VL-3B-Instruct的要求形成输入。
   * [ ] （徐）把Qwen2.5-VL-3B-Instruct上传到服务器，修改配置文件中模型和数据集路径。
2. SFT部分
   * [ ] （脱）修改`reinforced_cot/finetune/sft.py`，适配Qwen2.5-VL-3B-Instruct（注意使用4-bit量化减小显存）。
   * [ ] （脱）运行SFT，得到初步损失收敛曲线，并保存权重。
   * [ ] （脱）写模型推理逻辑，适配Qwen2.5-VL-3B-Instruct模型。
   * [ ] （周）根据推理接口和数据集，书写模型在制定测试集上的评测流程，参照GQA评测标准。
3. GRPO部分
   * [ ] （徐）设计基于规则的、适配CoT的回报函数。
   * [ ] （徐）写GRPO的policy model。
   * [ ] （徐）加载SFT模型并与当前策略模型计算KL损失，融合基于规则的回报。
   * [ ] （徐）实现GRPO训练流程并初步得到损失收敛曲线，保存权重。
   * [ ] （周）评测SFT和GRPO模型，负责分析模型输出的区别。

### 协作注意事项
- 格式化程序统一选择`black-formatter`，且配置为：
```json
"[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter"
  },
  "black-formatter.args": [
        "--line-length",
        "120"
    ],
```
- Git commit书写规范参考[这篇文章](https://blog.csdn.net/2301_79602429/article/details/145437838?spm=1011.2124.3001.6209)