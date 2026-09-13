# MVTec Grounding格式使用说明

## 一、代码修改总结

### 1. `data/load_mvtec_data.py` 的修改

**新增功能：**
- ✅ `extract_bbox_from_mask()`: 从mask图像提取bbox坐标
- ✅ `convert_to_grounding_format()`: 将MVTec样本转换为Grounding格式
- ✅ `get_all_grounding_samples()`: 获取所有Grounding格式样本
- ✅ `save_grounding_dataset()`: 保存Grounding格式数据集到JSON

**关键特性：**
- 自动从mask图像提取bbox（`[x1, y1, x2, y2]`格式）
- 将对话转换为Grounding格式（assistant输出JSON bbox）
- 保留原始对话内容，只修改assistant回复部分

### 2. `qwen3_vl_8b.py` 的修改

**新增功能：**
- ✅ `smart_resize()`: 智能resize图像，满足ViT patch要求
- ✅ `scale_bbox()`: bbox坐标缩放
- ✅ `MVTecQwenGroundingDataset`: Grounding格式数据集类
- ✅ `parse_grounding_output()`: 解析模型输出的bbox JSON
- ✅ 冻结ViT支持（`--freeze_vit`）
- ✅ Grounding格式支持（`--use_grounding_format`）

**训练配置优化：**
- 默认冻结视觉编码器（`freeze_vit=True`）
- LoRA rank调整为8（Grounding任务推荐）
- 支持smart_resize和bbox缩放

---

## 二、使用方法

### 1. 查看和转换数据格式

**查看数据统计：**
```bash
cd /data2/zlt/anomaly_detection_llm
python data/load_mvtec_data.py --mode view
```

**转换为Grounding格式：**
```bash
# 转换测试集
python data/load_mvtec_data.py --mode convert --dataset_mode test --output_path datasets/mvtec_grounding_test.json

# 转换训练集
python data/load_mvtec_data.py --mode convert --dataset_mode train --output_path datasets/mvtec_grounding_train.json
```

### 2. 训练Grounding模型

**基本训练命令：**
```bash
python qwen3_vl_8b.py \
    --mode train \
    --model_name Qwen/Qwen2-VL-7B-Instruct \
    --dataset_root /data2/zlt/anomaly_detection_llm/datasets/mvtec_anomaly_detection \
    --conversation_json_path /data2/zlt/anomaly_detection_llm/datasets/mvtec_zero_shot.json \
    --use_grounding_format \
    --freeze_vit \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 32 \
    --learning_rate 1e-4 \
    --max_image_size 1024 \
    --factor 28 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_epochs 1 \
    --output_dir ./outputs/qwen2vl_grounding_mvtec
```

**关键参数说明：**
- `--use_grounding_format`: 使用Grounding格式（必需）
- `--freeze_vit`: 冻结视觉编码器（推荐）
- `--lora_r 8`: LoRA rank（Grounding任务推荐8）
- `--max_image_size 1024`: 图像最大边长
- `--factor 28`: 图像尺寸必须是28的倍数（满足ViT patch要求）

### 3. 推理测试

```bash
python qwen3_vl_8b.py \
    --mode inference \
    --model_path ./outputs/qwen2vl_grounding_mvtec/final_model \
    --image_path /path/to/test_image.jpg \
    --prompt "Locate the anomaly region in this image and output the bbox coordinates in JSON format." \
    --max_image_size 1024 \
    --factor 28
```

---

## 三、数据格式说明

### Grounding格式示例

```json
{
  "id": "bottle_broken_large_000.png",
  "image": "bottle/test/broken_large/000.png",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nLocate the anomaly region in this image and output the bbox coordinates in JSON format."
    },
    {
      "from": "gpt",
      "value": "{\"bbox_2d\": [100, 200, 400, 350], \"label\": \"anomaly\", \"defect_type\": \"broken_large\", \"class\": \"bottle\"}"
    }
  ],
  "metadata": {
    "anomaly": true,
    "class": "bottle",
    "defect_type": "broken_large",
    "bbox": [100, 200, 400, 350],
    "source": "mvtec_anomaly_detection"
  }
}
```

### 正常样本格式

```json
{
  "id": "bottle_good_000.png",
  "image": "bottle/test/good/000.png",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nLocate the anomaly region in this image and output the bbox coordinates in JSON format. If no anomaly is found, output {\"bbox_2d\": null, \"label\": \"normal\"}."
    },
    {
      "from": "gpt",
      "value": "{\"bbox_2d\": null, \"label\": \"normal\"}"
    }
  ],
  "metadata": {
    "anomaly": false,
    "class": "bottle",
    "defect_type": "good",
    "bbox": null,
    "source": "mvtec_anomaly_detection"
  }
}
```

---

## 四、关键实现细节

### 1. Bbox提取

从mask图像提取bbox的算法：
```python
def extract_bbox_from_mask(mask_path):
    mask = Image.open(mask_path).convert("L")
    mask_array = np.array(mask) > 0
    
    rows = np.any(mask_array, axis=1)
    cols = np.any(mask_array, axis=0)
    
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    
    return [x1, y1, x2, y2]
```

### 2. Smart Resize

确保图像尺寸满足ViT patch要求：
```python
def smart_resize(image, max_size=1024, factor=28):
    w, h = image.size
    scale = min(max_size / max(w, h), 1.0)
    new_w = int(w * scale / factor) * factor
    new_h = int(h * scale / factor) * factor
    return image.resize((new_w, new_h))
```

### 3. Bbox缩放

图像resize后，bbox必须同步缩放：
```python
def scale_bbox(bbox, scale_factor):
    x1, y1, x2, y2 = bbox
    scale_x, scale_y = scale_factor
    return [
        int(x1 * scale_x),
        int(y1 * scale_y),
        int(x2 * scale_x),
        int(y2 * scale_y)
    ]
```

---

## 五、训练配置建议

### 推荐配置（基于Qwen2.5-VL Grounding教程）

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `freeze_vit` | `True` | 冻结视觉编码器，节省资源 |
| `lora_r` | `8` | Grounding任务推荐较小rank |
| `lora_alpha` | `32` | 通常为rank的4倍 |
| `learning_rate` | `1e-4` | 可以尝试1e-4到5e-4 |
| `max_image_size` | `1024` | 图像最大边长 |
| `factor` | `28` | 满足ViT patch要求 |
| `batch_size` | `1` | 根据GPU显存调整 |
| `gradient_accumulation_steps` | `16` | 相当于batch_size=16 |

### 训练流程

1. **数据准备**：
   ```bash
   python data/load_mvtec_data.py --mode convert --dataset_mode test
   ```

2. **开始训练**：
   ```bash
   python qwen3_vl_8b.py --mode train [参数...]
   ```

3. **验证模型**：
   ```bash
   python qwen3_vl_8b.py --mode inference [参数...]
   ```

---

## 六、输出解析

### 解析模型输出

模型输出示例：
```
The anomaly region is located at: {"bbox_2d": [341, 258, 397, 360], "label": "anomaly"}
```

解析函数会提取JSON部分：
```python
bbox_data = parse_grounding_output(response)
# 返回: {"bbox_2d": [341, 258, 397, 360], "label": "anomaly"}
```

### 坐标映射

推理时，模型输出的bbox是基于resize后的图像尺寸。如果需要映射回原始图像：
```python
# 反向缩放
inv_scale_x = 1.0 / scale_factor[0]
inv_scale_y = 1.0 / scale_factor[1]
original_bbox = [
    int(bbox[0] * inv_scale_x),
    int(bbox[1] * inv_scale_y),
    int(bbox[2] * inv_scale_x),
    int(bbox[3] * inv_scale_y)
]
```

---

## 七、常见问题

### Q1: Bbox坐标偏移

**原因：** 图像resize和bbox缩放不一致

**解决：**
- ✅ 确保使用`smart_resize`和`scale_bbox`
- ✅ 使用相同的缩放因子
- ✅ 检查transformers库版本（使用最新版本）

### Q2: 输出格式混乱

**原因：** 训练数据中assistant回复包含额外文字

**解决：**
- ✅ 确保assistant回复只包含JSON
- ✅ 使用`parse_grounding_output`提取JSON部分
- ✅ 在prompt中明确要求"只输出JSON"

### Q3: 图像尺寸不满足要求

**原因：** 图像尺寸不是factor的倍数

**解决：**
- ✅ 使用`smart_resize`自动处理
- ✅ 确保factor=28（Qwen2-VL的要求）

---

## 八、下一步

1. **数据验证**：运行转换脚本，检查Grounding格式数据
2. **小规模测试**：先用少量数据测试训练流程
3. **完整训练**：使用全部数据训练模型
4. **评估验证**：在测试集上评估bbox准确性

---

## 九、参考

- [Qwen2.5-VL Grounding微调教程](./Qwen2.5-VL_Grounding微调教程.md)
- [Qwen2.5-VL官方文档](https://qwenlm.github.io/blog/qwen2.5-vl/)
- [ms-swift微调工具](https://github.com/modelscope/swift)
