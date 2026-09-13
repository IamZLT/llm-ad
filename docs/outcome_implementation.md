# 两图五步主线：Outcome GRPO 实现与运行说明

本实现保留“理解—对比—定位—验证—输出”。默认仍为 **正常参考图 + 检测图 + H 的区域文本信息**。H 不作为第三张 RGB 图输入。新增代码使用独立入口 `train_outcome.py`，不改变旧 `train.py`、旧奖励或正在运行的实验。

## 研究主线如何表述

正常参考图提供正常外观依据，冻结视觉编码器产生参考条件下的差异先验 H。H 被转换为少量、可被否定的候选区域，供模型决定比较的重点。模型依次生成理解、对比、候选定位、验证与最终检测结果。最终分类与定位质量产生终态奖励，通过策略梯度优化整条生成序列。

- 理解：识别对象与相关正常结构。
- 对比：指出参考与检测图中具体的差异或一致性。
- 定位：提出待验证区域，允许 null，也允许在 H 之外提出候选。
- 验证：根据可见图像证据保留、修正、拒绝候选或发现其他区域。
- 输出：最终异常类别与全图坐标框。

这是**结构化生成 + 终态奖励**。五个标签提供任务组织形式，并不等于五步均有可靠的过程监督；目前也不能证明每句话忠实于模型的决策依据。两图版本的验证利用已有全图证据，不声称调用了新的视觉工具。需要通过固定数据与 checkpoint 的 H 干预实验检验 H 的实际效用。

## 新实现与旧实现的区别

| 项目 | 新主线 |
|---|---|
| 输入图像 | 2 张，与原主线相同 |
| H | 最多 3 个连通区域，含 bbox、峰值、均值和面积；允许空 |
| 五步流程 | 保留，每步简短，不设解释最低字数 |
| ground | JSON `{"candidate_bbox_2d": ...}`，候选不是最终标签 |
| verify | JSON `{"action": "keep/refine/reject/discover/none", "evidence": "简短依据"}` |
| answer | JSON `is_anomaly/bbox_2d/description` |
| 任务解析 | 独立于五段协议；准确 bbox 不会因缺少解释而失去任务分 |
| 终态奖励 | 最终结构非法或分类错误 -1；正常正确 +1；异常正确按 IoU |
| 协议奖励 | 严格协议有效时 +0.05，无字数奖励；可在配置中设 0 |
| 候选/方向/H 贴合奖励 | 移除 |
| advantage | 一条输出一个 `R - mean(R_group)`，默认不除组内 std |
| 策略损失 | 有效生成 token 的 PPO clipped surrogate + KL，逐轨迹平均再组平均 |
| 采样 | 显式 T=1、top_p=1、top_k=0、无重复惩罚，匹配 raw-policy logprob |
| 停止与 padding | 保留真实 EOS 或完成 `</answer>` 的 token，丢掉后续 batch padding |
| 预算 | 按尝试的 group 计数，零优势组照常记录并消耗预算，不无限重采样 |
| 评估 | `None` 表示完整数据；dev/test 命名空间分离；保存逐样本 JSONL |

任务标签仍为原数据扫描器生成的**所有缺陷的单个外包框**，不是多实例检测或像素分割。异常样本缺少合法 GT、正常样本带异常 GT 时直接报错。正常参考禁止与检测图同一文件。

格式协议奖励只评结构，不能证明 evidence 的语义正确。协议奖励 0.05 在任务分极近时仍可能影响排序，应记录并在协议稳定后做权重 0 的对照。

## H 的边界与校准

候选框从 patch 边缘计算，修复了单 patch、细条区域零宽/零高的问题。保留原始响应幅度，明确告诉模型它不是异常概率。默认 `relative_threshold: 0.7` 只是未校准的候选提议规则：平坦 H 返回空；非平坦正常图仍可能产生候选，须允许模型拒绝。

`raw_threshold` 可设固定原始幅度门槛。它本身也不等于已校准；本次没有声称完成正常数据校准。正式阈值需要使用训练划分内独立正常校准图确定，并报告图像级误报；不能使用 dev/test 标签挑阈值再报告无偏测试效果。跨域 MVTec 的正常分布可能与 VisA 不同。

## 文件与配置

- `outcome/protocol.py`：短五步 prompt、唯一 JSON 字段解析、任务/协议分离、终态奖励。
- `outcome/inputs.py`：H 区域化、双图输入、可选原图 ROI、统一几何、禁止静默截断。
- `outcome/policy.py`：结束位置、生成、统一 advantage、old/ref/new logprob 与 LoRA 更新。
- `outcome/engine.py`：有限训练预算、每次尝试日志、完整评估与按尺度/类别统计。
- `train_outcome.py`：训练、评估、预测共用同一输入/输出路径。
- `scripts/smoke_outcome.py`：真实模型的短集成检查，不保存权重；使用人工 advantage 检查反向传播，不代表性能提升。
- `tests/test_outcome.py`：奖励、解析、padding、坐标和循环/评估回归测试。
- `configs/qwen35_08b_outcome.yaml`：0.8B 双图通路检查。
- `configs/qwen35_2b_outcome.yaml`：**2B 双图五步主线**。
- `configs/qwen35_2b_outcome_roi.yaml`：独立的第三图 ROI 消融，默认主线不使用。

## 运行顺序

以下命令在 `/data2/zlt/anomaly_detection_llm` 执行，选择空闲 GPU；不复用旧输出目录。环境使用项目已有 Python 依赖。

1. 基础测试：

```bash
/home/zlt/miniconda3/bin/python -m pytest tests/test_protocol.py tests/test_outcome.py -q
```

2. 先测新协议下 base 模型，不直接接续旧 reward 的 checkpoint。下面默认完整 VisA dev；它才是后续 RL 的可比基线。

```bash
CUDA_VISIBLE_DEVICES=0 /home/zlt/miniconda3/bin/python train_outcome.py \
  --config configs/qwen35_2b_outcome.yaml --mode eval --split dev
```

若只想先检查流程，添加 `--eval-limit 8`；输出会记录样本数，8 张不能作为正式性能结论。完整 dev 目前预期为 240，取决于数据与配置，运行会保存实际样本数。

3. 短训练试运行（有初始和最终的 8 张诊断评估）：

```bash
CUDA_VISIBLE_DEVICES=0 /home/zlt/miniconda3/bin/python train_outcome.py \
  --config configs/qwen35_2b_outcome.yaml --mode train \
  --max-attempts 16 --eval-limit 8
```

检查 `train/task_valid_rate`、`train/zero_advantage_group`、`train/truncation_rate`、`optimizer/logprob_max_error` 和梯度。若大量组全为无效输出，先缩短协议或准备经过图像核验的 SFT 冷启动，不用扩大格式奖励补救。

4. 完整主线训练：

```bash
CUDA_VISIBLE_DEVICES=0 /home/zlt/miniconda3/bin/python train_outcome.py \
  --config configs/qwen35_2b_outcome.yaml --mode train
```

默认每张图 G=8、学习率 1e-6、一次策略更新、两轮数据对应的尝试预算、最大生成 512 tokens。512 是上限，不要求写满；通过日志比较实际生成长度。默认不自动跑 MVTec test，完成模型选择后显式执行最终 test。

5. 评估保存的 LoRA（将路径替换为真实输出）：

```bash
CUDA_VISIBLE_DEVICES=0 /home/zlt/miniconda3/bin/python train_outcome.py \
  --config configs/qwen35_2b_outcome.yaml --mode eval --split dev \
  --adapter outputs/train/qwen35_2b_outcome/实际运行目录/adapter_final
```

将 `--split dev` 改为 `--split test` 即完整最终测试（不传 `--eval-limit`）。

6. 双图预测：

```bash
CUDA_VISIBLE_DEVICES=0 /home/zlt/miniconda3/bin/python train_outcome.py \
  --config configs/qwen35_2b_outcome.yaml --mode predict \
  --adapter outputs/train/qwen35_2b_outcome/实际运行目录/adapter_final \
  --reference /绝对路径/正常参考.png --image /绝对路径/检测图.png --class-name bottle
```

返回 0–1000 全图框及原图像素框。旧 Flask/demo 仍属于旧入口，不能用它们评判新模型的双图+H 性能。

## SFT reference 的语义

`outcome.sft_adapter: null`：从配置的基座新建 RL LoRA，关闭 RL adapter 得到冻结 base reference。

若提供经过核验的 SFT adapter 路径，先将它合并到基座，再挂新的 RL LoRA。此时关闭 RL adapter 得到 SFT reference，不会误回到原始 base。评估该 RL adapter 必须使用相同配置与 SFT adapter。未在本次任务中生成 SFT 数据或执行正式 SFT/RL。

`--adapter` 仅供评估/预测，不代表恢复 optimizer。本实现暂不支持断点恢复优化器、分布式训练、任意多轮工具轨迹；遇到多 GPU 配置会显式报错，不默默以错误梯度训练。旧 DDP 路径保持原样。

## 第三图与消融

第三图是原始检测图的高清 ROI，**不是 H 热力图**。固定取 H 排名最高候选的原图裁剪，每边扩展候选宽/高的 25%；无候选时保持两图。正常参考全图始终保留，不盲目裁同坐标参考，因为当前未注册。

ROI 使用独立配置，新增视觉 token 与延迟均记录。当前为固定候选的复查器：输入 ROI 后再生成五步文本，**不声称模型先生成候选再调用裁剪工具**。若这项消融没有可靠收益，主线继续使用两图。

两图主线通过官方视觉塔的一次联合前向，同时捕获指定层 pre-merger 特征计算 H，并获得 merger 缓存；H 的 NN 与融合公式保持不变。避免手工逐图执行视觉塔造成的模型/形状相关 BF16 数值差异。三图分支在得到 H 候选后使用官方视觉塔再联合编码三图，然后共享缓存给整组生成和 logprob；这项额外预处理耗时计入 ROI 成本。

建议先固定 checkpoint 做 `outcome.prior.condition: real / none / shuffled` 的配对评估，其他设置完全相同。`shuffled` 是确定性的空间打乱 H，保留幅度分布，不是另一张真实图的 H；不要把它描述为完整的“错误参考图”实验。先关闭 ROI 检验 H 文本提示，再单独加入 ROI；否则 H 消融同时改变了额外图像观察。

## 日志怎么读

- `train/*` 的横轴是尝试次数，包含零优势组。
- `optimizer/*` 的横轴是实际 optimizer 更新次数。
- 首版每组只更新一次，更新前 `ratio≈1`、`clip_fraction≈0`、组内中心化后的 PG 标量约为 0 都是预期行为；其梯度仍可非零。不要把 PG 标量接近 0 当作模型没有学习，主要检查梯度、参数更新与固定 dev 指标。该设置下 PPO ratio clipping 通常不激活，梯度范数裁剪仍生效。
- `dev/*` 使用完整/显式限量的 dev；`dev_final/*` 与 `test_final/*` 分开。
- `normal_fpr`：正常图中明确预测异常的比例。非法类别不会被算作正确正常，另外报告 `invalid_decision_rate` 和 `normal_correct_rate`。
- `anomaly_recall`：异常图明确预测异常的比例；最终框非法时定位 IoU 为 0。
- `anomaly_gated_miou` 与 `acc_at_05`：异常图上，任务字段合法且判断异常的最终框质量；无效或误判正常计 0。
- `task_valid_rate` 与 `protocol_valid_rate`：任务可解析率与完整五步结构率分开。
- `prompt_tokens / visual_tokens / prior_hint_tokens / mean_new_tokens`：用于量化两图主线及 ROI 的真实 token 成本。
- 每个 run 保存配置、代码 hash、split manifest、逐样本/逐轨迹预测。

实际性能是否提升，需要用同一数据划分与参考图完成正式基线和 RL 对比。本次代码测试、缓存一致性与反向传播检查不构成缺陷检测收益证据。

当前 SFT / 2B·4B RL 的实测数字（MVTec 200 张分层 eval、训练 rollout、机制是否激活）写在 [multibox_实验结果.md](multibox_实验结果.md)。
