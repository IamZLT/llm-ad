# GT 框粒度两难：union 大框 vs 连通域小框，论文怎么解？

> 问题：一张图有多个缺陷时——
> - **GT 用 union 大框**：模型只检出其中一部分缺陷时，IoU 被大框里的空白稀释，很吃亏；
> - **GT 用逐连通域小框**：缺陷又小又密集时，GT 变成一堆离散小框，模型很难把数量数对，
>   `count_reward` / Hungarian 集合奖励持续惩罚，训练信号噪声大。
>
> 本文档调研论文界的解法，并对照本仓库现状给出可落地建议。

---

## 1. 论文界的五种解法

### 1.1 GT 构建时先合并：形态学膨胀 + 连通域（AD-FM，AAAI'26）

我们的 reward 设计来源 AD-FM 在 **GT 构建阶段**就正面处理了这个问题：

> "We first apply **morphological dilation** to the raw anomaly masks for **merging fragmented regions**,
> then use connected component analysis to extract minimal enclosing boxes."
> —— AD-FM, arXiv:2508.04175, Sec.4 Bounding Box Generation

即：**先对 mask 做形态学膨胀，把邻近的碎片区域连成一片，再提连通域外接框**。
小而近的缺陷被合并成一个合理的 GT 框，数量奖励和 Hungarian 匹配都在"合并后的语义粒度"上进行。
（kernel 尺寸在补充材料，正文未给出；通常取图像短边的 1%~2%。）

**注意：本仓库当前明确不做这一步** —— `data/scan.py::extract_targets_from_mask` 的 docstring 写明
"No dilation, no proximity merging, no union-box fallback"，遵循的是 Datumaro 逐连通域惯例。

### 1.2 指标层吸收粒度差：mask 空间 IoU（AD-Copilot / MMAD-BBox）

AD-Copilot 提出 **BBox-Mask IoU** 的动机**正是**粒度不一致问题：

> "To handle the irregular shapes and **disconnected fragments** typical of industrial defects...
> converts both predicted and ground-truth boxes into binary masks and computes IoU in the mask space.
> This **alleviates bias caused by differing output granularities**."
> —— AD-Copilot, arXiv:2603.13779

思路：预测框集合和 GT 框集合**各自栅格化成一张二值 mask**，在 mask 空间算 IoU。
模型出一个大框盖住三个小 GT 框、或出三个小框，只要覆盖的区域一致，分数就一样——
**配对无关、粒度无关**。同时他们用**低 IoU 阈值**（0.1/0.2/0.3/0.5）报准确率，
因为异常定位天然比通用检测难。

本仓库的 `mask_iou`（reward 主项，权重 0.5）就是这个指标，**这一条我们已经做了**。

### 1.3 按区域等权评估：PRO / AUPIMO（经典 pixel-AD 的答案）

- **MVTec 官方论文**（Bergmann et al., IJCV 2020）提出 **PRO（per-region overlap）**：
  把 GT mask 分解为连通域，**每个区域单独算召回再平均**，"gives equal importance to each
  anomalous region irrespective of whether it is large or subtle defect"。
- **AUPIMO**（arXiv:2401.01984）同样以连通域为 region 单位计算 per-region 召回/FPR。

经典异常检测社区的做法是：**永远不 union，逐区域评估**，大缺陷检出不能弥补小缺陷漏检。
这对应我们 eval 里的 `recall_at_01/03/05`（逐 GT 连通域是否被命中）。

### 1.4 密度聚类定粒度：DBSCAN（eps 控制合并半径）

- 东京大学 IAD referring 工作（E495）：把整片异常 mask 用 **DBSCAN** 拆成单个异常区域，
  eps 决定"多近算同一个缺陷"。
- PCB 缺陷检测（SCITEPRESS'21）：对二值输出的正像素做 DBSCAN（eps=2），
  每个 cluster 出一个框再与 GT 比较。

DBSCAN 是"膨胀合并"的更可控版本：合并半径是显式超参，不怕膨胀核把框撑大。

### 1.5 放弃区分：COCO `iscrowd`（检测社区的答案）

COCO 对**超过 10–15 个紧密聚集的同类实例**不再逐个标注，而是标成单个 crowd 区域，
**评估时整片忽略**（既不算 TP 也不算 FP）。
对应到我们的场景：如果某类样本缺陷极端密集（如 VisA 里 63 个连通域的 mask），
可以把它们标记为 "crowd-like"，不参与 count reward，只用 mask_iou 评估。

---

## 2. 本仓库现状对照

| 环节 | 现状 | 粒度敏感性 |
|---|---|---|
| GT 构建（`extract_targets_from_mask`） | 逐连通域，min_area=5，**不膨胀不合并** | —— |
| `mask_iou`（reward 主项 ×0.5） | mask 空间 IoU | **粒度无关**（已对冲） |
| `union_iou`（eval 参考项） | union 大框 IoU | 仅供 benchmark 对照 |
| `count_reward`（×0.2） | \|N−M\| 精确计数 | **对 GT 粒度最敏感** |
| `set_localization_reward`（×0.3） | Hungarian 后 sum/max(N,M) | 分母含 M，敏感 |
| `detection F1@50/75`、`recall@` | 逐框匹配 | 敏感 |

结论：**union 吃亏的问题我们已经通过"不用 union 当监督"规避了**；
但"小而近 → GT 离散"的问题目前**完全暴露**在 `count_reward` 和 `set_localization_reward` 上。

---

## 3. 可落地的改进方案（按改动量排序）

### 方案 A：GT 构建加可选的形态学合并（AD-FM 正统做法，推荐）✅ 已实现

**AD-FM 源码调研结论（2026-09-13）**：AD-FM 无官方开源仓库（AAAI 页面无代码链接，
arXiv v1 PDF 未附 supplementary），膨胀核尺寸等细节写在不可获取的补充材料里。
因此本实现是对论文文字描述（"morphological dilation → connected component analysis →
minimal enclosing boxes"）的忠实复刻，关键假设如下：

- 膨胀核：椭圆形，`k = max(3, round(min(H, W) * merge_kernel_ratio))` 取奇数，
  `ratio=0.01` 在 700–1024px 图上 ≈ 7–10px，可桥接 ≤k−1 px 的碎片间隙；
- 顺序：**先在原始 mask 上去噪**（面积 < `min_contour_area` 的连通域删除），再膨胀——
  防止噪声点被膨胀放大成假缺陷（论文未说明，这是本实现的额外保护）；
- 框从**膨胀后**的连通域提取（与论文一致，框会比原始缺陷略大 ~k px）；
- union box 始终从**原始 mask** 计算，不受合并影响，benchmark 口径不变。

实现位置与配置：

- `data/scan.py::extract_targets_from_mask(mask_path, min_contour_area=5, merge_kernel_ratio=0.0)`
  —— `ratio=0` 完全等价于旧行为；
- `scan_visa` / `scan_mvtec` / `load_prior_split` 透传，配置项 `data.merge_kernel_ratio`；
- 已在 `configs/qwen35_2b_outcome_multibox.yaml` / `qwen35_4b_outcome_multibox.yaml`
  设置 `merge_kernel_ratio: 0.01`；
- 测试：`tests/test_scan_merge.py`（7 例：合并/不合并/远距离/去噪/union 不变/空 mask）。

**VisA-train 全量统计（1200 异常图）**：

| ratio | mean | p50 | p95 | p99 | max | multi(≥2) 占比 |
|---|---|---|---|---|---|---|
| 0（旧） | 2.68 | 1 | 9 | 20 | 63 | 44.8% |
| 0.01 | 1.80 | 1 | 5 | 10 | 28 | 36.2% |
| 0.02 | 1.56 | 1 | 4 | 6 | 17 | 31.4% |

`DEFAULT_MAX_BOXES = 16` 在 ratio=0.01 下仍覆盖 p99（10），**无需调整**。

**注意事项**：
- 合并改变 GT 粒度 → 训练/评估口径同时变，**与旧 run 的指标不可直接比**，
  建议重跑一次 baseline eval 作为新对照；
- SFT target、`count_reward`、`set_localization_reward`、eval 分层全部读
  `meta['component_bboxes']`，自动跟随新粒度，无需改动；
- 正在运行的旧训练进程不受影响（数据已加载），下次启动生效。

### 方案 B：双粒度 reward，取 max（改动最小，不动 GT）

保留现有 GT 不变，在 `score_output` 里额外构造一份"膨胀合并后的 GT"，
对两种粒度各算一遍 `count_reward` 和 `set_localization_reward`，**取两者最大值**：

```python
count_r = max(count_reward(N, M_raw), count_reward(N, M_merged))
set_r   = max(set_localization_reward(pred, gt_raw),
              set_localization_reward(pred, gt_merged))
```

语义："不管你想分开报还是合并报，只要有一种合理粒度下是对的就行"。
与 mask_iou 的粒度无关性形成一致哲学，且不需要重统计数据、不动 eval。

### 方案 C：软化 count_reward

- 微小连通域（面积 < 图像面积 0.05% 之类）不计入 M；
- 或对 GT 先做一遍 dilation 再数连通域个数，用"合并后数量"当 M。
- 成本最低，但治标不治本（set reward 仍敏感）。

### 方案 D：调权重，让粒度无关项主导

把 `iou_weight`(mask_iou) 从 0.5 调高、`count_weight` 从 0.2 调低。
零代码改动，但放弃了数量监督信号，多框场景的定位精度可能退化。

### 评估侧建议（无论选哪个方案）

- 以 `mask_miou` 为主指标（粒度无关），`recall@0.3` / `det F1@0.5` 为辅；
- 继续按 `component_bin`（single/multi）分层报，专门盯 multi 桶的变化；
- 可增加"合并后 GT"口径的 `recall@` / `count_error`，双口径并列。

---

## 4. 参考

- AD-FM: *Multimodal LLMs for Anomaly Detection via Multi-Stage Reasoning and Fine-Grained Reward
  Optimization*, AAAI 2026, arXiv:2508.04175（形态学膨胀合并碎片 + 连通域提框）
- AD-Copilot: arXiv:2603.13779（BBox-Mask IoU，粒度无关指标；低 IoU 阈值 0.1–0.5）
- MVTec AD 官方论文: Bergmann et al., IJCV 2020（PRO：逐连通域等权评估）
- AUPIMO: arXiv:2401.01984（per-region 指标）
- COCO: Lin et al., ECCV 2014（iscrowd：密集实例标记后评估忽略）
- 本仓库：`data/scan.py::extract_targets_from_mask`、`outcome/protocol_multibox.py::score_output`、
  `outcome/metrics.py::mask_iou / component_metrics`
- 相关文档：`docs/multibox_handling.md`（多框输出/奖励/评估全景）
