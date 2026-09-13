# Qwen2.5-VL Grounding任务微调教程

## 一、Grounding任务简介

**Grounding（定位/指向）** 是指给定一张图像 + 一段自然语言指令（如"请在图中定位 camera，并输出框"），模型理解指令后输出目标物体的边界框（bbox）。

与传统检测器（如YOLO）相比，Grounding的优势：
- 更灵活，可接受开放词汇与文字描述
- 支持自然语言指令，无需预定义类别
- 可以处理复杂的定位任务

---

## 二、数据集格式

### 1. 标准JSON/JSONL格式

Qwen2.5-VL使用**对话格式**的数据，每个样本包含图像和对话内容。

#### 基本结构

```json
{
  "id": "sample_001",
  "image": "path/to/image.jpg",  // 或使用 "images": ["path1.jpg", "path2.jpg"] 支持多图
  "conversations": [
    {
      "role": "system",  // 可选
      "content": [
        {
          "type": "text",
          "text": "You are a grounding assistant..."
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "image",
          "image": "path/to/image.jpg"
        },
        {
          "type": "text",
          "text": "<image>\nLocate house in this image and output the bbox coordinates in JSON format."
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "{\"bbox_2d\": [x1, y1, x2, y2]}"
        }
      ]
    }
  ]
}
```

### 2. 简化格式（兼容LLaVA格式）

也可以使用简化的格式，类似于Anomaly-OneVision使用的格式：

```json
{
  "id": "sample_001",
  "image": "path/to/image.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nLocate house in this image and output the bbox coordinates in JSON format."
    },
    {
      "from": "gpt",
      "value": "{\"bbox_2d\": [x1, y1, x2, y2]}"
    }
  ]
}
```

### 3. Grounding输出格式

Assistant的回复应该包含JSON格式的边界框信息：

**单个目标：**
```json
{
  "bbox_2d": [100, 200, 400, 350],
  "label": "car"
}
```

**多个目标：**
```json
[
  {
    "bbox_2d": [100, 200, 400, 350],
    "label": "car"
  },
  {
    "bbox_2d": [500, 220, 620, 330],
    "label": "truck"
  }
]
```

**带子标签：**
```json
[
  {
    "bbox_2d": [341, 258, 397, 360],
    "label": "motorcyclist",
    "sub_label": "not wearing helmet"
  },
  {
    "bbox_2d": [5, 235, 63, 320],
    "label": "motorcyclist",
    "sub_label": "wearing helmet"
  }
]
```

### 4. 坐标格式要求

**关键点：**
- ✅ 使用**绝对坐标**（像素坐标），而非归一化坐标
- ✅ 格式为 `[x1, y1, x2, y2]`（左上角和右下角坐标）
- ✅ 坐标基于**原始图像尺寸**

**坐标转换：**
- 如果原始数据是 `[x, y, width, height]` 格式，需要转换为 `[x1, y1, x2, y2]`
- 转换公式：`x2 = x + width`, `y2 = y + height`

---

## 三、数据预处理

### 1. 图像尺寸调整

**要求：**
- 图像会被resize到模型输入尺寸（如最大边长限制）
- 宽高需要是某个factor的倍数（如28的倍数），以满足ViT patch划分要求
- 使用 `smart_resize` 或类似函数保持宽高比

**示例：**
```python
def smart_resize(image, max_size=1024, factor=28):
    """智能resize，确保尺寸是factor的倍数"""
    w, h = image.size
    scale = min(max_size / max(w, h), 1.0)
    new_w = int(w * scale / factor) * factor
    new_h = int(h * scale / factor) * factor
    return image.resize((new_w, new_h))
```

### 2. Bbox坐标缩放

**重要：** 当图像被resize后，bbox坐标必须同步缩放！

```python
def scale_bbox(bbox, original_size, new_size):
    """缩放bbox坐标"""
    scale_x = new_size[0] / original_size[0]
    scale_y = new_size[1] / original_size[1]
    
    x1, y1, x2, y2 = bbox
    new_bbox = [
        int(x1 * scale_x),
        int(y1 * scale_y),
        int(x2 * scale_x),
        int(y2 * scale_y)
    ]
    return new_bbox
```

### 3. 坐标格式转换

```python
def convert_xywh_to_xyxy(bbox):
    """从 [x, y, width, height] 转换为 [x1, y1, x2, y2]"""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]
```

---

## 四、微调训练

### 1. 使用ms-swift工具（推荐）

```bash
swift sft \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --dataset your_grounding_dataset \
  --train_type lora \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --freeze_vit true \
  --gradient_accumulation_steps 16 \
  --eval_steps 100 \
  --save_steps 100 \
  --save_total_limit 2 \
  --logging_steps 5 \
  --max_length 2048 \
  --output_dir output \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 4 \
  --dataset_num_proc 4
```

### 2. 关键参数说明

| 参数 | 说明 |
|------|------|
| `--freeze_vit true` | 冻结视觉编码器，只训练语言模型和融合层，节省资源 |
| `--train_type lora` | 使用LoRA参数高效微调 |
| `--lora_rank 8` | LoRA的rank大小 |
| `--lora_alpha 32` | LoRA的alpha参数 |
| `--target_modules all-linear` | 对所有线性层应用LoRA |
| `--gradient_accumulation_steps 16` | 梯度累积步数，相当于batch_size=16 |

### 3. 训练策略

**推荐配置：**
- ✅ 冻结视觉编码器（`freeze_vit true`）
- ✅ 使用LoRA微调语言模型
- ✅ 学习率：1e-4 到 5e-4
- ✅ Batch size：根据GPU显存调整，使用梯度累积

---

## 五、推理与验证

### 1. 推理格式

```python
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

processor = AutoProcessor.from_pretrained("your_model_path")
model = AutoModelForCausalLM.from_pretrained("your_model_path")

image = Image.open("test_image.jpg")
prompt = "<image>\nLocate all cars in this image and output the bbox coordinates in JSON format."

inputs = processor(text=[prompt], images=[image], return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512)
response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
```

### 2. 输出解析

```python
import json
import re

def parse_grounding_output(response):
    """解析模型输出的bbox JSON"""
    # 提取JSON部分
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        json_str = json_match.group()
        try:
            bbox_data = json.loads(json_str)
            return bbox_data
        except:
            pass
    
    # 如果是数组格式
    array_match = re.search(r'\[.*\]', response, re.DOTALL)
    if array_match:
        array_str = array_match.group()
        try:
            bbox_list = json.loads(array_str)
            return bbox_list
        except:
            pass
    
    return None
```

### 3. 可视化验证

```python
from PIL import ImageDraw

def visualize_bbox(image, bbox_data, output_path):
    """在图像上绘制bbox"""
    draw = ImageDraw.Draw(image)
    
    if isinstance(bbox_data, list):
        bboxes = bbox_data
    else:
        bboxes = [bbox_data]
    
    for item in bboxes:
        if "bbox_2d" in item:
            x1, y1, x2, y2 = item["bbox_2d"]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            if "label" in item:
                draw.text((x1, y1-20), item["label"], fill="red")
    
    image.save(output_path)
```

---

## 六、常见问题与解决方案

### 1. Bounding Box系统性偏移

**问题：** bbox位置整体偏上或偏移

**原因：**
- 坐标转换或图像resize不一致
- transformers库中RoPE或M-RoPE版本问题

**解决：**
- ✅ 使用新版本的transformers（2025年及以后版本）
- ✅ 严格按照模型提供的`smart_resize` + 坐标缩放公式处理数据
- ✅ 确保图像resize和bbox缩放使用相同的比例

### 2. 坐标与patch不对齐

**问题：** 坐标与模型内部patch划分不对齐

**原因：** 图像尺寸不满足模型对patch的要求（例如不能整除factor）

**解决：**
- ✅ Resize时确保宽高满足模型要求（如28的倍数）
- ✅ 使用`smart_resize`函数自动处理

### 3. 输出格式混乱

**问题：** 模型输出包含额外说明文字，不是纯JSON

**原因：** 训练数据中assistant响应包含多余文本

**解决：**
- ✅ 在prompt中明确要求"只输出JSON"或"只给bbox JSON"
- ✅ 训练数据中assistant回复只包含JSON，不要额外说明
- ✅ 使用后处理提取JSON部分

### 4. 坐标格式不一致

**问题：** 模型输出坐标格式与期望不符

**解决：**
- ✅ 统一使用`[x1, y1, x2, y2]`格式
- ✅ 在训练数据中保持一致
- ✅ 推理时进行格式验证和转换

---

## 七、完整示例：MVTec异常检测转Grounding格式

### 原始数据格式

```json
{
  "id": "mvtec_00000000",
  "image": "mvtec_anomaly_detection/capsule/test/poke/004.png",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nIs there any anomaly on the capsule shown in the image?"
    },
    {
      "from": "gpt",
      "value": "Yes, there is an anomaly on the capsule..."
    }
  ],
  "metadata": {
    "anomaly": true,
    "mask": "mvtec_anomaly_detection/capsule/ground_truth/poke/004_mask.png"
  }
}
```

### 转换为Grounding格式

```python
import json
from PIL import Image
import numpy as np

def convert_to_grounding_format(sample, mask_path, image_folder):
    """将MVTec数据转换为Grounding格式"""
    
    # 加载mask图像
    mask = Image.open(os.path.join(image_folder, mask_path)).convert("L")
    mask_array = np.array(mask) > 0
    
    # 找到bbox（简化版本，实际应该更精确）
    rows = np.any(mask_array, axis=1)
    cols = np.any(mask_array, axis=0)
    if rows.any() and cols.any():
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        bbox = [int(x1), int(y1), int(x2), int(y2)]
    else:
        return None  # 没有异常区域
    
    # 构建Grounding格式
    grounding_sample = {
        "id": sample["id"],
        "image": sample["image"],
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nLocate the anomaly region in this image and output the bbox coordinates in JSON format."
            },
            {
                "from": "gpt",
                "value": json.dumps({"bbox_2d": bbox, "label": "anomaly"})
            }
        ]
    }
    
    return grounding_sample
```

---

## 八、总结

### 关键要点

1. **数据格式**：
   - 使用对话格式（conversations）
   - Assistant回复包含JSON格式的bbox
   - 使用绝对坐标 `[x1, y1, x2, y2]`

2. **预处理**：
   - 图像resize要满足patch要求
   - bbox坐标必须同步缩放
   - 保持坐标格式一致性

3. **训练**：
   - 推荐冻结视觉编码器
   - 使用LoRA微调
   - 合理设置学习率和batch size

4. **验证**：
   - 解析JSON输出
   - 可视化bbox验证
   - 检查坐标准确性

### 参考资源

- [Qwen2.5-VL官方文档](https://qwenlm.github.io/blog/qwen2.5-vl/)
- [ms-swift微调工具](https://github.com/modelscope/swift)
- [Grounding任务示例](https://github.com/DavideNapolitano/Qwen-VL-Finetune)

---

## 九、与Anomaly-OneVision的对比

| 特性 | Anomaly-OneVision | Qwen2.5-VL Grounding |
|------|-------------------|---------------------|
| 任务类型 | 异常检测（Yes/No + 描述） | 目标定位（bbox输出） |
| 输出格式 | 文本描述 | JSON格式bbox |
| 数据格式 | conversations（from/value） | conversations（role/content）或from/value |
| 坐标 | 不需要 | 需要绝对坐标 |
| 图像处理 | 支持anyres | 需要smart_resize |
| 微调方式 | 全参数或部分参数 | 推荐LoRA + 冻结ViT |

**转换建议：**
- 可以将Anomaly-OneVision的异常检测任务扩展为Grounding任务
- 需要从mask图像中提取bbox
- 修改conversations格式，assistant输出改为JSON bbox
