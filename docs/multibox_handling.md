# 多框（multi-box）异常检测：本仓库做法与 VLM 论文对比

> 场景：一张检测图可能出现**多个互不连通的缺陷区域**，模型需要输出**一组** bbox，而不是一个框。
> 本文档回答三个问题：
> 1. 多框时我们**现在**是怎么处理的？
> 2. 主流 VLM 做检测时一般怎么处理多框？
> 3. 我们的方案处于什么位置，还能怎么改？

---

## 1. 本仓库当前做法（`outcome-multibox-v1`）

### 1.1 GT 怎么从 mask 变成"多个框"

- `data/scan.py::extract_targets_from_mask` 用 `cv2.findContours(RETR_EXTERNAL)` 把缺陷 mask 拆成**每个连通域一个外接框**。
- 同时保留一个 `bbox`（所有缺陷的 union box）供兼容旧指标。
- `max_boxes` 不是拍脑袋：用 **VisA-train** 的 `num_components` 高分位数冻结（`compute_max_boxes`），4B 配置里是 **16**（p95=9 / p99=20），评估集（MVTec）不参与这个统计。

### 1.2 模型输出协议

- `<ground>` 里给**候选框列表**：`candidate_bboxes_2d=[[x1,y1,x2,y2], ...]`（或 `[]`）。
- `<answer>` 里给**最终框列表**：`bboxes_2d`（异常）或 `[]`（正常）。
- 坐标是 **0–1000 归一化**，顺序就是生成顺序，**不做 NMS、不排序、不合并**。
- 解析器 `parse_output` 只校验：每框 4 个数、`x1<x2`、`y1<y2`、总数 ≤ `max_boxes`、JSON 合法。

### 1.3 训练奖励（`score_output`）——多框怎么打分

异常图（`is_anomaly=True` 且分类正确）时：

```
task = cls_weight
     + iou_weight   * mask_iou(pred_boxes, gt_components)      # 主定位项
     + count_weight * count_reward(N_pred, M_gt)               # 数量项
     + dense_weight * set_localization_reward(pred, gt)        # DCLR 稠密项
     + refine_weight* min(0, delta_refine)                     # 不许把候选改差
```

- **`mask_iou`（AD-Copilot BBox-Mask IoU）**：把预测框集合和 GT 连通域集合各自栅格化成二值 mask，在 mask 空间算 IoU。**不配对、不关心框数**，天然适合不规则/断裂缺陷。
- **`count_reward`（AD-FM eq.5）**：`|N−M|=0→1.0`，差 1→0.5，差 ≥2→−0.1。显式惩罚"数错框"。
- **`set_localization_reward`（DCLR 风格）**：预测框 × GT 框算 IoU 矩阵 → **Hungarian 一对一匹配** → `sum(matched IoU) / max(N, M)`。分母同时惩罚漏检和多检。
- **`delta_refine`**：`final` 的 set reward 减去 `candidate` 的 set reward，只保留负值，防止 refine 阶段把本来对的候选改坏。

正常图（`is_anomaly=False`）时：

```
task = normal_correct + focus_weight * focus_reward(N_candidate)
```

- **`focus_reward`（AD-FM eq.6）**：0 框→0，1 框→0.5，≥2 框→−0.1。鼓励"先定位一个可疑点，再拒绝"，而不是直接空输出。

### 1.4 评估指标（`evaluate_multibox.py`）

- **分类**：`anomaly_recall`、`normal_fpr`、`balanced_accuracy`。
- **定位**：
  - `mask_miou` / `union_miou`：mask 空间 / union box 的 IoU。
  - `matched_miou`、`set_iou`：Hungarian 匹配后的平均 IoU 与集合 IoU。
  - `det_precision/recall/f1_at_50/75`：把文本框当检测框，在 IoU 阈值下的单点 PR（VLM 没有置信度，所以不是 mAP）。
  - `count_error = |N_pred − M_gt|`。
- **H 先验对照**：`iou_h_top1/bestk`、`prior_component_recall_*`，看 region token 提供的候选本身有多好。
- **分层**：按缺陷大小（small/medium/large）和连通域数（single/multi）分别统计。

### 1.5 一句话总结我们的策略

> **输出端**：自回归生成一个变长框列表，无 NMS。
> **监督端**：mask-IoU（不管配对）+ Hungarian set reward（管配对）+ count reward（管数量）三路并行。

---

## 2. VLM 论文里多框一般怎么处理

### 2.1 自回归"坐标即文本"派（和我们同一路线）

| 工作 | 多框输出 | 训练目标 | 备注 |
|---|---|---|---|
| **Pix2Seq** (ICLR'22) | 把 `[x1,y1,x2,y2,class]` 序列化成 token 序列，一次生成多个对象 | 纯交叉熵，无检测头 | 证明"语言建模"就能做检测 |
| **Florence-2** (CVPR'24) | 坐标量化成 `<loc_0>`–`<loc_999>` 特殊 token，多框 = 多段 token | 标准 CE | 去掉 L1/GIoU 回归头，RefCOCO 反超带检测头方法 |
| **Qwen2.5-VL / Qwen3-VL** | JSON/纯文本坐标，归一化 [0,1000] | 坐标当普通文本 token，CE | 官方明确：**没有 YOLO/DETR 式框回归 loss** |
| **Shikra / Ferret / Kosmos-2** | 文本坐标或少量特殊 token | CE | 主要靠 SFT 数据规模 |

**共同点**：多框 = 变长序列，**不配对、不 NMS**，靠数据让模型学会"数个数 + 给坐标"。

### 2.2 集合预测 + 匈牙利匹配派（经典检测器思路）

| 工作 | 多框输出 | 训练目标 | 备注 |
|---|---|---|---|
| **DETR** | 固定数量 object query → 集合输出 | Hungarian 匹配 + L1/GIoU | 端到端、无 NMS |
| **Grounding DINO** | 开集检测，query + 文本条件 | Hungarian + contrastive | 不是 LLM，是专用检测器 |

**区别**：这类方法有**显式集合预测头**和**可微的 Hungarian loss**；VLM 把坐标离散化后，Hungarian 只能放在 **reward/评估** 里，不能直接反传。

### 2.3 工业异常检测里的多框奖励（我们直接对标的）

| 工作 | 多框处理 | 奖励设计 |
|---|---|---|
| **AD-FM** (AAAI'26) | 预测框集合 vs GT 框集合，Hungarian 匹配 | `r_loc = mean GIoU(matched) + α·r_count`；`r_count` 就是 1/0.5/−0.1；正常图用 `r_focus` |
| **AD-Copilot** (2026) | 提出 **BBox-Mask IoU**：预测框和 GT 框都转二值 mask 再算 IoU | RL 阶段 `R = λR_fmt + IoU(B_pred, M_gt)`，专治不规则/断裂缺陷 |

我们的 `mask_iou` + `count_reward` + `focus_reward` 基本就是 **AD-FM + AD-Copilot 的组合实现**。

---

## 3. 对比与取舍

| 维度 | 我们的做法 | 典型 VLM | 备注 |
|---|---|---|---|
| 输出形式 | JSON 数组 `bboxes_2d` | 文本坐标序列 / `<loc>` token | 我们更结构化，易解析 |
| 坐标空间 | 0–1000 归一化 | 0–1000 或原始像素 | 与 Qwen 官方一致 |
| 框数上限 | `max_boxes=16`（数据驱动） | 通常无硬上限或很大 | 防止无限生成 |
| 训练信号 | CE（SFT）+ RL reward | 纯 CE 或 CE+RL | 我们多了显式集合 reward |
| 配对方式 | 训练/评估都用 Hungarian | 评估用 Hungarian，训练一般不用 | 我们把匹配搬进了 reward |
| 重复框处理 | 不 NMS，靠 count reward 惩罚 | 通常也不 NMS | 重复框会拉低 `set_iou` 和 `count_reward` |
| 正常样本 | `focus_reward` 鼓励"定位后拒绝" | 一般只给分类奖励 | AD-FM 的关键设计 |

### 我们方案的优点

1. **对断裂缺陷友好**：`mask_iou` 不强迫一对一，一个长划痕被模型拆成两个框也不会被误判为全错。
2. **数量可学**：`count_reward` 直接告诉 GRPO"数错了"，比单纯 IoU 更敏感。
3. **冷启动稳定**：SFT 先学会输出合法多框 JSON，RL 再调精度，避免一开始 reward 全 0。

### 已知局限 / 可改进点

> **GT 粒度问题（union 吃亏 / 小近缺陷太离散）的专题调研见 `docs/gt_box_granularity.md`。**


1. **没有置信度**：所有框等权，无法像检测器那样做 soft-NMS 或按分数排序。
2. **Hungarian 不可微**：只能作为 RL reward，不能给 SFT 提供逐框梯度。
3. **`max_boxes=16` 是 VisA 统计出来的**：如果换更碎的数据集，需要重新跑 `compute_max_boxes`。
4. **refine 还没学会**：当前 `delta_refine` 基本为 0，说明 `<ground>` 和 `<answer>` 框几乎一样，后续可以加大 `refine_weight` 或给 candidate/final 设计不同噪声。

---

## 4. 参考

- `data/scan.py::extract_targets_from_mask` / `compute_max_boxes`
- `outcome/protocol_multibox.py::parse_output` / `score_output` / `count_reward` / `focus_reward` / `set_localization_reward`
- `outcome/metrics.py::mask_iou` / `component_metrics` / `detection_metrics` / `set_giou`
- `outcome/evaluate_multibox.py::make_record` / `evaluate`
- AD-FM: *Multimodal LLMs for Anomaly Detection via Multi-Stage Reasoning and Fine-Grained Reward Optimization*, AAAI 2026, arXiv:2508.04175
- AD-Copilot: *A Vision-Language Assistant for Industrial Anomaly Detection via Visual In-context Comparison*, arXiv:2603.13779
- Pix2Seq: *A Language Modeling Framework for Object Detection*, ICLR 2022, arXiv:2109.10852
- Florence-2: *Advancing a Unified Representation for a Variety of Vision Tasks*, CVPR 2024
- Qwen2.5-VL / Qwen3-VL Technical Reports
