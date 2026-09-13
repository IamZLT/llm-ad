# 旧单框 GRPO 模型（5002 端口）MVTec 评测结果分析

> 评测日期：2026-09-09
> 模型：`grpo_checkpoint-2650`
> 输出目录：`outputs/train/qwen35_2b_prior/qwen35_2b_visa2mvtec_20260905_210255/`
> 评测脚本：`scripts/eval_prior_grpo.py`

---

## 1. 模型背景

5002 端口对应 `configs/qwen35_2b_grpo.yaml`，是**较早的旧链路**：

- **无 SFT 预热**，直接 GRPO
- **文本 hint**：将 H（多层差异先验热力图）的 top-k 空间锚点写成 `<prior_hint>` 文本描述，塞进 prompt
- **单框输出**：`bbox_2d`（单个框），而非多框 `bboxes_2d`
- 冻结视觉塔 + LoRA（r=16）微调语言侧
- 训练数据：VisA 全量（异常+正常）→ 测试：MVTec

这条链路后来被「视觉 token + SFT 预热 + 多框 outcome」的新链路（5004 端口）取代。新链路截至 2026-09-13 的数字见 [multibox_实验结果.md](multibox_实验结果.md)。

---

## 2. 评测配置

- **数据集**：MVTec test
- **采样**：分层抽样，15 个类别 × 每类 6 正常 + 6 异常 = **180 张**（异常 90 / 正常 90）
- **设备**：单卡 RTX 3090（GPU 7）
- **方式**：加载 `grpo_checkpoint-2650` 的 LoRA adapter 后推理

> 说明：该 run 在训练中被中断，没有跑过最终的完整 MVTec test（`run_final_mvtec_eval` 只在训练全部结束后执行）。本次评测由独立脚本补跑。

---

## 3. 总体结果

| 指标 | 数值 | 说明 |
|---|---|---|
| 识别准确率 `rec_acc` | **0.500** | 平衡数据下全判异常也能得 50% |
| JSON 解析率 | 1.000 | 输出格式完全合规 |
| 异常命中 IoU@0.3 | 0.578 | 异常样本中 52/90 命中 |
| 异常 mIoU | 0.355 | 定位精度 |
| acc@0.1 / 0.3 / 0.5 | 0.800 / 0.578 / 0.322 | 分阈值识别+定位联合 |
| 分尺寸 IoU（小/中/大） | 0.278 / 0.315 / 0.442 | 缺陷越大定位越好 |

---

## 4. 每类别结果

| 类别 | 样本数 | 异常召回 | 正常拒识 | 异常 mIoU |
|---|---|---|---|---|
| bottle | 12 | 1.000 | 0.000 | 0.480 |
| cable | 12 | 1.000 | 0.000 | 0.315 |
| capsule | 12 | 1.000 | 0.000 | 0.313 |
| carpet | 12 | 1.000 | 0.000 | 0.616 |
| grid | 12 | 1.000 | 0.000 | 0.459 |
| hazelnut | 12 | 1.000 | 0.000 | 0.560 |
| leather | 12 | 1.000 | 0.000 | 0.251 |
| metal_nut | 12 | 1.000 | 0.000 | 0.113 |
| pill | 12 | 1.000 | 0.000 | 0.158 |
| screw | 12 | 1.000 | 0.000 | 0.167 |
| tile | 12 | 1.000 | 0.000 | 0.431 |
| toothbrush | 12 | 1.000 | 0.000 | 0.411 |
| transistor | 12 | 1.000 | 0.000 | 0.308 |
| wood | 12 | 1.000 | 0.000 | 0.654 |
| zipper | 12 | 1.000 | 0.000 | 0.091 |

**关键观察：所有 15 个类别的异常召回都是 1.000，正常拒识全是 0.000。**

---

## 5. 核心诊断

### 5.1 严重的「全判异常」偏置

```
异常样本：预测为 anomaly = 90 / 90   （召回 100%）
正常样本：预测为 anomaly =  0 / 90   （特异度 0%）
```

模型把 **180 张图全部判成了异常**。

- `rec_acc = 0.5` 是「异常:正常 = 1:1」平衡数据下全判异常也能拿 50% 的假象，**不反映真实判别能力**。
- 判别能力（正常/异常二分类）实质上是**失效**的。

### 5.2 定位能力尚可但分化明显

- 异常 mIoU 0.355，大缺陷（如 carpet 0.616、wood 0.654、hazelnut 0.560）表现不错。
- 但小/细长缺陷很差：`zipper 0.091`、`metal_nut 0.113`、`pill 0.158`、`screw 0.167`。
- 说明模型能「看到 H 高响应就给个框」，但对细小结构缺陷的定位精度不足。

---

## 6. 结论与启示

1. **判别失效是这条链路的根本问题**：模型学会了「对 H 高响应区域给框」，但没有学会「判定到底是不是真异常」。这大概率是文本 hint 过于低层、无法让模型建立「正常变化 vs 真实缺陷」的判别信号所致。

2. **`rec_acc=0.5` 是陷阱指标**：在 1:1 平衡评测下会被「全判异常」策略作弊。后续评测应同时看 **异常召回 + 正常特异度（TNR）** 两个维度，或用 `balanced_accuracy = (recall + TNR) / 2` 作为主指标。

3. **这正是切换到新链路的原因**：
   - 文本 hint → **视觉 region token**（H 引导的 region 对比 token，由 `RegionAdapter` 生成）
   - 无 SFT → **SFT 预热**（`region_sft_multibox`，先对齐输出格式与判别能力）
   - 单框 → **多框 outcome**（`bboxes_2d`，用 AD-Copilot BBox-Mask IoU 作为定位 reward）

---

## 7. 相关论文 / 对比方法性能水平

> 以下数据均来自各论文原文或公开 MVTec AD 排行榜，用于衡量「别的论文一般能达到什么水平」。
> **重要提示**：不同方法的评测口径不同（AUROC vs 准确率 vs mIoU、不同数据集、不同设定），**不能直接逐行比大小**，需结合标注说明。

### 7.1 传统无监督 AD 方法（MVTec AD，15 类，image-level AUROC）

传统方法把异常检测当作「阈值无关的排序问题」，用 AUROC 衡量，已接近饱和：

| 方法 | 会议/年份 | I-AUROC (图级) | P-AUROC (像素级) | 说明 |
|---|---|---|---|---|
| Dinomaly2 | - | **99.9%** | - | 已饱和，无法再区分方法优劣 |
| SimpleNet | CVPR 2023 | 99.6% | ~98.1% | 当前图级 SOTA |
| FastFlow | 2023 | 99.4% | - | 2D 归一化流 |
| PatchCore | CVPR 2022 | 99.1% | 98.1~98.6% | 最常用基线 |
| EfficientAD | WACV 2024 | 99.1% | ~97.8% | 实时推理 |
| Reverse Distillation | CVPR 2022 | 98.5% | - | 教师-学生 |
| CFlow-AD | WACV 2022 | 98.3% | - | 归一化流 |
| DRAEM | ICCV 2021 | 98.0% | - | 合成异常训练 |
| PaDiM | ICPR 2021 | 97.9% | ~97.3% | 高斯建模 |

> 结论：传统方法在图级 AUROC 上普遍 **> 97%**，判别能力（正常/异常二分类）非常强。这是「判别失效」问题的参照系。

### 7.2 基于 LVLM / MLLM 的异常检测方法

这是与本项目（LLM-based AD）最相关的对比对象：

| 方法 | 类型 | MVTec-AD 图级 | 定位指标 | 备注 |
|---|---|---|---|---|
| **AnomalyGPT** | LVLM（无监督训练） | I-AUC 97.4% / P-AUC 93.1% / acc 93.3% | 像素级 AUC | 需在目标域正常样本上训练 |
| **AnomalyGPT**（1-shot，VisA→MVTec） | LVLM（少样本） | I-AUC 94.1% / P-AUC 95.3% / acc 86.1% | 像素级 AUC | 跨域少样本 |
| **Anomaly-OV** | MLLM（零样本） | I-AUC **94.0%**（zero-shot） | - | Look-Twice Feature Matching；9 数据集平均 88.6% |
| **AD-Copilot** (7B) | MLLM（视觉 in-context 对比） | MMAD 平均 acc 78.71% | MMAD-BBox mIoU **25.3%** / IoU@0.5 21% | 引入 Comparison Encoder |
| **AD-Copilot-Thinking** (7B) | MLLM | MMAD 平均 acc **82.29%** | 同上 | 加入思考链后提升 |
| WinCLIP | CLIP（零样本） | I-AUC 91.8% | - | 纯预训练 CLIP |
| AnomalyCLIP | CLIP（零样本） | I-AUC 91.5% | - | 面向异常微调 prompt |

**AD-Copilot 在 MMAD 基准上的细分对比**（7 个子任务平均 accuracy）：

| 模型 | 平均 acc |
|---|---|
| Human (expert) | 86.65% |
| GPT-4o | 74.9% |
| Qwen2.5-VL (7B，基线) | 72.19% |
| AD-Copilot (7B) | 78.71% |
| AD-Copilot-Thinking (7B) | **82.29%** |

### 7.3 本文模型（5002 端口）所处水平

| 模型 | 判别（正常/异常） | 定位 |
|---|---|---|
| 本文旧单框 GRPO（`ckpt-2650`） | rec_acc **0.500**（召回 100% / 特异度 0%） | MVTec 异常 mIoU **0.355** |

对比结论：

1. **判别能力远低于所有参照系**：传统方法图级 AUROC >97%、Anomaly-OV zero-shot 94.0%、AD-Copilot 82.3%，而本文模型「全判异常」导致平衡准确率仅 50%，**判别基本失效**。
2. **定位 mIoU（0.355）表面上高于 AD-Copilot 的 MMAD-BBox mIoU（25.3%）**，但需注意：
   - 数据集不同：本文是 MVTec test，AD-Copilot 是 MMAD-BBox（跨品类工业图，难度更高）；
   - 本文在「把正常也框出来」的情况下测的是异常子集 mIoU，存在样本选择偏差；
   - 两者不可直接等同，只能说本文在「已检出异常样本」上的框精度尚可，但代价是零特异度。
3. **核心差距不在定位而在判别**：传统方法和强 MLLM 都先解决了「是不是异常」，再谈「在哪里」；本文旧链路跳过了前者。

---

## 8. 附：复现命令

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/eval_prior_grpo.py \
  --config configs/qwen35_2b_grpo.yaml \
  --ckpt outputs/train/qwen35_2b_prior/qwen35_2b_visa2mvtec_20260905_210255/grpo_checkpoint-2650 \
  --per-class 6 \
  --out outputs/train/qwen35_2b_prior/qwen35_2b_visa2mvtec_20260905_210255/eval_steps/mvtec_test_ckpt2650.json
```

输出文件：
- `eval_steps/mvtec_test_ckpt2650.json`（汇总指标）
- `eval_steps/mvtec_test_ckpt2650_records.json`（逐样本明细）
