# 多GPU训练指南

## 方法1：使用 torchrun（推荐）

### 基本用法

```bash
# 使用2个GPU
torchrun --nproc_per_node=2 qwen3_vl_8b.py \
    --mode train \
    --model_name Qwen/Qwen3-VL-8B-Instruct \
    --dataset_root /data2/zlt/anomaly_detection_llm/datasets/mvtec_anomaly_detection \
    --conversation_json_path /data2/zlt/anomaly_detection_llm/datasets/mvtec_zero_shot.json \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --output_dir ./outputs/qwen3_vl_grounding_mvtec

# 使用4个GPU
torchrun --nproc_per_node=4 qwen3_vl_8b.py \
    --mode train \
    [其他参数...]

# 使用8个GPU
torchrun --nproc_per_node=8 qwen3_vl_8b.py \
    --mode train \
    [其他参数...]
```

### 指定GPU

```bash
# 使用GPU 0,1,2,3
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 qwen3_vl_8b.py \
    --mode train \
    [其他参数...]
```

## 方法2：使用 DeepSpeed（推荐用于大模型）

DeepSpeed可以更高效地利用多GPU，并且可以节省内存。

### 安装DeepSpeed

```bash
pip install deepspeed
```

### 使用DeepSpeed Zero-2（推荐）

```bash
# 使用DeepSpeed Zero-2配置
torchrun --nproc_per_node=4 qwen3_vl_8b.py \
    --mode train \
    --deepspeed scripts/zero2.json \
    --model_name Qwen/Qwen3-VL-8B-Instruct \
    --dataset_root /data2/zlt/anomaly_detection_llm/datasets/mvtec_anomaly_detection \
    --conversation_json_path /data2/zlt/anomaly_detection_llm/datasets/mvtec_zero_shot.json \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --output_dir ./outputs/qwen3_vl_grounding_mvtec
```

### 使用DeepSpeed Zero-3（内存最省，但可能更慢）

```bash
# 使用DeepSpeed Zero-3配置（适合超大模型）
torchrun --nproc_per_node=4 qwen3_vl_8b.py \
    --mode train \
    --deepspeed scripts/zero3.json \
    [其他参数...]
```

## 方法3：使用 accelerate launch

```bash
# 首先配置accelerate
accelerate config

# 然后启动训练
accelerate launch qwen3_vl_8b.py \
    --mode train \
    [其他参数...]
```

## Batch Size 计算

多GPU训练时，总的有效batch size计算：

```
总有效batch size = per_device_batch_size × gradient_accumulation_steps × num_gpus
```

例如：
- `per_device_batch_size=1`
- `gradient_accumulation_steps=8`
- `num_gpus=4`
- **总有效batch size = 1 × 8 × 4 = 32**

## 示例脚本

### 2 GPU训练

```bash
#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 qwen3_vl_8b.py \
    --mode train \
    --model_name Qwen/Qwen3-VL-8B-Instruct \
    --dataset_root /data2/zlt/anomaly_detection_llm/datasets/mvtec_anomaly_detection \
    --conversation_json_path /data2/zlt/anomaly_detection_llm/datasets/mvtec_zero_shot.json \
    --use_grounding_format \
    --freeze_vit \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 32 \
    --learning_rate 1e-4 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_epochs 1 \
    --output_dir ./outputs/qwen3_vl_grounding_mvtec
```

### 4 GPU + DeepSpeed训练

```bash
#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 qwen3_vl_8b.py \
    --mode train \
    --deepspeed scripts/zero2.json \
    --model_name Qwen/Qwen3-VL-8B-Instruct \
    --dataset_root /data2/zlt/anomaly_detection_llm/datasets/mvtec_anomaly_detection \
    --conversation_json_path /data2/zlt/anomaly_detection_llm/datasets/mvtec_zero_shot.json \
    --use_grounding_format \
    --freeze_vit \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 32 \
    --learning_rate 1e-4 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_epochs 1 \
    --output_dir ./outputs/qwen3_vl_grounding_mvtec
```

## 注意事项

1. **内存使用**：
   - 使用DeepSpeed Zero-2/3可以显著减少内存使用
   - 如果遇到OOM，可以减小`batch_size`或增加`gradient_accumulation_steps`

2. **学习率调整**：
   - 多GPU训练时，通常不需要调整学习率
   - 如果使用更大的总batch size，可能需要适当增加学习率

3. **日志和保存**：
   - 所有GPU的日志都会保存到TensorBoard
   - Checkpoint只会在主进程（rank 0）保存

4. **性能优化**：
   - 使用DeepSpeed通常比DDP更快
   - 确保数据加载器有足够的workers（`--num_workers`）

## 检查GPU使用情况

```bash
# 实时监控GPU使用
watch -n 1 nvidia-smi

# 或者
nvidia-smi -l 1
```

## 常见问题

### Q: 如何知道使用了多少个GPU？

**A:** 训练开始时会显示GPU配置信息，或者查看nvidia-smi。

### Q: DeepSpeed和DDP有什么区别？

**A:** 
- **DDP**: 每个GPU保存完整模型副本，内存使用较高
- **DeepSpeed**: 可以分片存储模型参数和优化器状态，内存使用更低

### Q: 如何选择Zero-2还是Zero-3？

**A:**
- **Zero-2**: 平衡性能和内存，推荐用于大多数情况
- **Zero-3**: 内存最省，但通信开销更大，适合超大模型

### Q: 训练速度没有提升？

**A:** 
- 检查数据加载是否是瓶颈（增加`num_workers`）
- 确保batch size足够大
- 使用DeepSpeed可能比DDP更快
