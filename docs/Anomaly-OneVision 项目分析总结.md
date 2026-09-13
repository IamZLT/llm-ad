# Anomaly-OneVision 项目分析总结

## 1. 数据集加载方式

### 1.1 数据配置格式
项目使用 **YAML配置文件** (`data/datasets.yaml`) 来管理多个数据集：

```yaml
datasets:
  - json_path: "/path/to/dataset1.json"
    sampling_strategy: "all"  # 或 "first:1000", "end:3000", "random:10%"
  - json_path: "/path/to/dataset2.json"
    sampling_strategy: "all"
```

### 1.2 数据加载代码位置
- **主要加载代码**: `llava/train/train.py` 中的 `LazySupervisedDataset` 类（第957-1016行）
- **数据格式**: JSON文件，每个样本包含：
  - `id`: 样本ID
  - `image`: 图像路径（相对于image_folder）
  - `conversations`: 对话列表，格式为 `[{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]`
  - `metadata`: 元数据（可选，包含anomaly标签等）

### 1.3 采样策略
支持多种采样策略：
- `"all"`: 使用所有数据
- `"first:N"`: 使用前N个样本
- `"end:N"`: 使用后N个样本
- `"random:N"` 或 `"random:N%"`: 随机采样N个或N%的样本

## 2. 训练集和测试集设置

### 2.1 训练集
- **配置方式**: 通过 `--data_path` 参数指定YAML配置文件
- **示例**: `--data_path /data/02/jiacong/data/our_datasets2.yaml`
- **图像文件夹**: 通过 `--image_folder` 指定图像根目录
- **训练脚本**: `scripts/train/finetune_anomalyov_7b.sh`

### 2.2 测试集
- **配置方式**: 通过JSON文件指定测试数据
- **评估脚本**: `scripts/eval/test_detection.py` 和 `scripts/eval/test_reasoning.py`
- **参数**: `--bench_json` 指定测试JSON文件路径
- **数据格式**: 与训练数据类似，但用于评估

### 2.3 数据预处理
- **图像处理**: 支持多种aspect ratio（square, pad, anyres等）
- **对话预处理**: 在 `preprocess` 函数中处理对话格式，添加特殊token（IMAGE_TOKEN等）
- **标签mask**: 使用 `IGNORE_INDEX` 来mask掉不需要计算loss的部分（用户输入部分）

## 3. Loss函数

### 3.1 主要Loss
项目使用 **标准的语言模型CrossEntropyLoss**：
- **位置**: 在语言模型的forward函数中计算（`llava/model/language_model/modeling_llama.py` 第1332行）
- **实现**:
```python
loss_fct = CrossEntropyLoss()
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()
loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
```

### 3.2 Loss计算特点
1. **只计算assistant回复部分**: 通过mask机制，只对模型生成的回复部分计算loss
2. **忽略用户输入**: 用户输入部分使用 `IGNORE_INDEX` 标记，不参与loss计算
3. **标准next-token prediction**: 预测下一个token，标准的自回归语言模型训练方式

### 3.3 预处理中的Mask机制
在 `preprocess` 函数中（`llava/train/train.py`）：
- 用户输入部分：`target += [IGNORE_INDEX] * len(encode_id)`
- Assistant回复部分：`target += encode_id`（正常计算loss）

## 4. 评估指标

### 4.1 检测任务指标（test_detection.py）
位置：`scripts/eval/calc_accuracy.py`

使用的指标：
- **Accuracy**: 准确率
- **Precision**: 精确率
- **Recall**: 召回率
- **F1 Score**: F1分数

### 4.2 评估方式
1. **二分类判断**: 通过解析模型输出中的 "Yes" 或 "No" 来判断是否有异常
   ```python
   pred = 1 if 'Yes' in responses[index] or 'yes' in responses[index] else 0
   ```

2. **按类别评估**: 对每个类别（class）分别计算指标
   ```python
   for cls in class_name:
       # 计算该类别下的accuracy, precision, recall, f1
   ```

3. **平均指标**: 计算所有类别的平均指标

### 4.3 推理任务评估（test_reasoning.py）
- 使用多轮对话进行评估
- 支持历史对话上下文
- 评估模型对异常的解释和推理能力

## 5. 训练配置要点

### 5.1 关键训练参数（来自finetune_anomalyov_7b.sh）
- `--num_train_epochs 1`: 训练1个epoch
- `--per_device_train_batch_size 1`: 每个设备batch size为1
- `--gradient_accumulation_steps 4`: 梯度累积4步
- `--learning_rate 1e-5`: 学习率
- `--mm_tunable_parts="mm_mlp_adapter,mm_language_model"`: 可训练部分
- `--mm_vision_tower_lr=1e-6`: 视觉编码器学习率（较小）
- `--bf16 True`: 使用bfloat16精度
- `--gradient_checkpointing True`: 梯度检查点以节省内存

### 5.2 模型架构
- **基础模型**: Qwen2-7B-Instruct
- **视觉编码器**: SigLIP (google/siglip-so400m-patch14-384)
- **多模态连接器**: MLP projector (mm_mlp_adapter)
- **异常检测专家**: AnomalyOV（在推理时使用）

## 6. 数据格式示例

### 训练数据格式
```json
{
  "id": "mvtec_00000000",
  "image": "mvtec_anomaly_detection/capsule/test/poke/004.png",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nWhat does the image depict?"
    },
    {
      "from": "gpt",
      "value": "The image depicts a pharmaceutical capsule..."
    },
    {
      "from": "human",
      "value": "Is there any anomaly on the capsule shown in the image?"
    },
    {
      "from": "gpt",
      "value": "Yes, there is an anomaly on the capsule..."
    }
  ],
  "metadata": {
    "anomaly": true,
    "source": "mvtec_anomaly_detection",
    "mask": "mvtec_anomaly_detection/capsule/ground_truth/poke/004_mask.png",
    "category": "industrial",
    "shot": "zero_shot",
    "target": "2D"
  }
}
```

## 7. 关键代码文件

1. **数据加载**: `llava/train/train.py` - `LazySupervisedDataset` 类
2. **数据预处理**: `llava/train/train.py` - `preprocess` 函数
3. **训练器**: `llava/train/llava_trainer.py` - `LLaVATrainer` 类
4. **Loss计算**: `llava/model/language_model/modeling_llama.py` - `LlamaForCausalLM.forward`
5. **评估指标**: `scripts/eval/calc_accuracy.py` - `cal_metrics` 函数
6. **检测评估**: `scripts/eval/test_detection.py`
7. **推理评估**: `scripts/eval/test_reasoning.py`

## 总结

- **数据集**: 使用YAML配置管理多个JSON数据集，支持灵活的采样策略
- **训练/测试**: 训练集通过YAML配置，测试集通过JSON文件指定
- **Loss**: 标准CrossEntropyLoss，只计算assistant回复部分
- **指标**: Accuracy, Precision, Recall, F1 Score，通过解析"Yes"/"No"进行二分类评估
