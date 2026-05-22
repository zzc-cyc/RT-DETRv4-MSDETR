# RT-DETRv4-S 官方训练方法完整解析

> 基于 `configs/rtv4/rtv4_hgnetv2_s_coco.yml` 及其完整 include 链逆向分析。官方日志来源：`logs/RTv4-S-hgnet.log`。

---

## 一、模型架构

```
输入 (640×640)
  │
  ▼
HGNetV2-B0 (backbone)
  │ return_idx [1,2,3] → 多尺度特征 [256, 512, 1024]
  │ use_lab: True, freeze_at: -1, freeze_norm: False
  ▼
HybridEncoder (RepNCSPELAN4 + CSP)
  │ depth_mult=0.34, expansion=0.5, hidden_dim=256
  │ distill_teacher_dim=768（接收 DINOv3 teacher 特征做蒸馏）
  ▼
DFINETransformer (D-FINE decoder, 3 layers)
  │ eval_idx=-1（取最后一层输出）, activation=silu
  ▼
检测头 → bbox (分布预测 + FDR) + class (MAL)
```

| 指标 | 值 |
|------|-----|
| 参数量 | ~10.5M |
| 输入尺寸 | 640×640 |
| 类别数 | 80 (COCO) |

---

## 二、训练超参数总览

### 优化器

| 参数 | 值 | 说明 |
|------|-----|------|
| 类型 | **AdamW** | |
| 主学习率 | **0.0004** | encoder / decoder / head |
| backbone 学习率 | **0.0002** | 骨干网络独立 LR，更保守 |
| weight decay | **0.0001** | L2 正则 |
| norm/bias 层 weight decay | **0** | 不对归一化层和偏置做衰减 |
| betas | [0.9, 0.999] | |
| 梯度裁剪 | max_norm=**0.1** | |

### 训练规模

| 参数 | 值 |
|------|-----|
| 总 epoch | **132** |
| 训练 batch size | **32**（4 GPU × 8 per GPU，DDP） |
| 验证 batch size | **64** |
| num_workers | **4** |
| AMP | **开启**（GradScaler 混合精度） |
| SyncBN | **开启**（跨 GPU 同步 BatchNorm） |
| EMA | **开启**（decay=0.9999, warmup=1000 iterations） |
| iters/epoch | 3696（COCO train2017 ≈ 118K / 32） |

---

## 三、学习率调度：FlatCosine

```
LR
│
│  warmup (2000 iter)
│   ┌──┐   flat (epoch 0-63)           cosine decay (64-119)        no_aug (120-131)
│  ╱    │   LR = init_lr 不变               LR ↘ min_lr               LR = min_lr
│ ╱     │   ┌──────────────────────┐  ╲                              ┌──────────┐
│╱      │   │                      │   ╲                            │          │
└───────┴───┴──────────────────────┴────╲───────────────────────────┴──────────┴──→ epoch
        0                       64                              120          132
```

| 参数 | 值 | 说明 |
|------|-----|------|
| 调度器类型 | `flatcosine` | 平顶余弦 |
| warmup_iter | **2000** | 前 2000 次迭代线性 warmup |
| flat_epoch | **64** | 前 64 个 epoch LR 保持 init_lr 不变 |
| lr_gamma | **0.5** | min_lr = init_lr × 0.5 |
| no_aug_epoch | **12** | 最后 12 个 epoch LR 降至 min_lr |

**设计意图**：flat 阶段用大 LR 充分探索，避免过早陷入局部最优；no_aug 阶段降 LR + 关增强做精细收敛。

---

## 四、数据增强策略

### 增强算子列表

| 算子 | 参数 | 说明 |
|------|------|------|
| Mosaic | output_size=320, rotation=10°, translation=[0.1,0.1], scaling=[0.5,1.5], prob=1.0 | 4 图拼接 |
| RandomPhotometricDistort | p=0.5 | 亮度/对比度/饱和度/色调扰动 |
| RandomZoomOut | fill=0 | 随机缩小画布 |
| RandomIoUCrop | p=0.8 | 基于 IoU 的随机裁剪 |
| SanitizeBoundingBoxes | min_size=1 | 过滤无效 bbox |
| RandomHorizontalFlip | — | 水平翻转 |
| Resize | [640, 640] | 统一尺寸 |
| MixUp | prob=0.5 | 两图混合（仅 collate_fn 生效） |

### 四阶段增强调度

```
policy.epoch = [4, 64, 120]
mixup_epochs = [4, 64]
stop_epoch   = 120
```

| 阶段 | epoch | 强度 | 具体内容 |
|------|-------|------|---------|
| **基础增强** | 0-3 | 轻 | HorizontalFlip + Resize。**无** Mosaic/ZoomOut/IoUCrop/PhotometricDistort/MixUp |
| **全量强增强** | 4-63 | 重 | 全部 8 种增强算子，MixUp 开启。这是让模型学会鲁棒表征的核心阶段 |
| **增强收缩** | 64-119 | 中 | 部分算子下线或概率降低（policy 字段控制） |
| **无增强精调** | 120-131 | 无 | 仅 Resize（base_size=640 单尺度），MixUp 和 Mosaic 等全部关闭。stop_epoch 在此生效 |

**epoch 4 分界线**：开启强增强后训练 loss 会上升、AP 可能短暂回落，这是正常行为，官方日志中也观察到相同现象。

---

## 五、损失函数 (RTv4Criterion)

| 损失项 | 权重 | 类型 | 作用 |
|--------|------|------|------|
| `loss_mal` | 1 | 分类 | Mutual Attraction Loss：用 IoU 质量分数加权分类 |
| `loss_bbox` | 5 | 回归 | L1 bbox 坐标损失 |
| `loss_giou` | 2 | 回归 | GIoU 损失 |
| `loss_fgl` | 0.15 | 回归 | Fine-Grained Localization（D-FINE 核心创新） |
| `loss_ddf` | 1.5 | 回归 | DDF 分布解码损失 |
| `loss_distill` | **5** | 蒸馏 | DINOv3 ViT-B encoder 特征蒸馏 |

总损失公式（简化）：

```
L_total = 1×L_mal + 5×L_bbox + 2×L_giou + 0.15×L_fgl + 1.5×L_ddf + 5×L_distill
```

### 自适应蒸馏权重

```yaml
distill_adaptive_params:
  enabled: True
  rho: 11       # 目标 encoder 梯度占比 11%
  delta: 1      # 容差 ±1%
  default_weight: 20  # 异常时的回退权重
```

根据 encoder 梯度回传比例动态调整 `loss_distill` 权重，使蒸馏信号保持在合理强度。

---

## 六、DINOv3 Teacher 蒸馏机制

```
每个训练 iteration:
  ┌─────────────────────────────────────────────┐
  │  images (batch)                              │
  │    │                                         │
  │    ├─→ HGNetV2 (student) → encoder_features │
  │    │                                         │
  │    └─→ DINOv3 ViT-B (teacher, no_grad)      │
  │         → teacher_encoder_features            │
  │              │                                │
  │              ▼                                │
  │    loss_distill = 1 - cos_sim(student, teacher) │
  │    (仅 encoder 输出层做特征对齐)                │
  └─────────────────────────────────────────────┘
```

| 属性 | 值 |
|------|-----|
| Teacher 模型 | DINOv3 ViT-B (patch_size=16) |
| Teacher 特征维度 | 768 |
| 蒸馏位置 | HybridEncoder 输出（encoder 特征层） |
| 蒸馏损失类型 | Cosine Similarity Loss |
| 计算开销 | **每 batch 额外 ~3 秒**（约占单步耗时的 85%） |
| 梯度 | Teacher 输出 detach，不反传 |

---

## 七、EMA 与两阶段训练

### EMA 配置

| 参数 | 值 |
|------|-----|
| decay | 0.9999 |
| warmup | 1000 iterations |

### 两阶段设计

```
stage 1 (epoch 0-119)          stage 2 (epoch 120-131)
┌─────────────────────┐        ┌─────────────────────┐
│ • 完整增强 + 多尺度   │  120  │ • 仅单尺度增强         │
│ • 保存 best_stg1.pth │ ────→ │ • 从 best_stg1.pth     │
│ • EMA decay=0.9999  │        │   刷新 EMA             │
│                     │        │ • 保存 best_stg2.pth   │
└─────────────────────┘        └─────────────────────┘
```

`det_solver.py` 中的关键逻辑：

```python
# epoch == stop_epoch (120) 时刷新 EMA
if epoch == self.train_dataloader.collate_fn.stop_epoch:
    self.load_resume_state('best_stg1.pth')    # 加载 stage1 最佳权重
    self.ema.decay = ema_restart_decay          # 重置 EMA decay
```

---

## 八、Checkpoint 与输出

| 项 | 说明 |
|----|------|
| `last.pth` | 最新 checkpoint（每 epoch，仅 stage1） |
| `checkpointXXXX.pth` | 按 `checkpoint_freq` 间隔保存 |
| `best_stg1.pth` | stage1 最佳 AP 权重 |
| `best_stg2.pth` | stage2 最佳 AP 权重（epoch ≥ 120） |
| `log.txt` | JSONL 格式，每 epoch 一行完整指标 |

---

## 九、训练全程 Timeline

```
Epoch  0: warmup，基础增强，DINOv3 teacher 蒸馏开始
Epoch  4: ★ 全量强增强 + MixUp 开启（loss 上升/AP 波动是正常的）
Epoch 12: ★ 训练中断恢复 — s/it 从 3.5 骤降至 0.5（疑似硬件迁移），蒸馏权重重置至默认值 5.0，teacher 仍在正常运行
Epoch 64: LR 开始 cosine 衰减，增强策略收缩
Epoch 120: ★ stop_epoch — 关闭多尺度，EMA 从 best_stg1.pth 刷新
Epoch 132: 训练结束，最终 AP ~0.498
```

---

## 十、官方日志异常：epoch 12 训练中断恢复

官方日志在 epoch 12 出现以下突变：

| 指标 | epoch 0-11 | epoch 12+ |
|------|-----------|----------|
| s/it | ~3.5 | ~0.5 |
| loss_distill（原始值） | ~5.6 | ~0.81 |
| loss_distill（加权后） | ~690（weight=123.77） | ~4.1（weight=5→8.89） |
| avg encoder grad | ~10.2% | ~7.1%（逐步恢复到~11.9%） |
| 蒸馏权重 | ~123.77（自适应稳定） | 5.0→逐步恢复至 128.55 |
| 单 epoch 耗时 | ~3.6 小时 | ~0.5 小时 |

### 实际发生的情况

**训练在 epoch 12 处中断并恢复，但 teacher 并未丢失。** 证据：

1. **`loss_distill` 始终非零**：epoch 12 的原始 distill loss 为 ~0.81（对比 epoch 11 的 ~5.6），仍在正常计算
2. **`avg encoder grad` 持续非零**：epoch 12 为 7.12%，epoch 13 为 6.98%，后续逐步恢复到 11.9%。若 teacher 丢失，该值应为 0%
3. **teacher 模型独立加载**：DINOv3 ViT-B 作为预训练模型，每次初始化时从自身权重文件加载，不依赖 checkpoint 的 `state_dict()`

### 真正发生的变化

- **蒸馏权重重置**：自适应蒸馏的当前权重（123.77）未保存在 checkpoint 中，resume 后从默认值 5.0 重新开始自适应调节（epoch 12-19 逐步恢复至 128.55）
- **硬件变更（最可能）**：s/it 从 3.5 骤降至 0.5，7 倍差异远超 teacher 前向开销所能解释（teacher 前向约占总计算量的 15-20%），更可能是训练从较慢 GPU 迁移至更快 GPU（如 A100/H100）

### 影响

无论速度变化原因如何，蒸馏始终在运行。v4-S 最终 AP 达到 0.498，说明该训练配置有效。

---

## 十一、关键代码入口

| 文件 | 作用 |
|------|------|
| `engine/solver/det_solver.py` | 训练主循环（fit/val/state_dict） |
| `engine/solver/det_engine.py` | `train_one_epoch` 函数，含 teacher forward |
| `engine/rtv4/rtv4_criterion.py` | 损失函数（MAL/FGL/DDF/蒸馏） |
| `engine/optim/lr_scheduler.py` | FlatCosine LR 调度器 |
| `engine/core/_config.py` | 配置基类 |
| `configs/rtv4/rtv4_hgnetv2_s_coco.yml` | S 模型配置入口 |

---

## 十二、与本地的差异对照

| 参数 | 官方 | 本地 (batch 8) |
|------|------|---------------|
| train batch size | 32 (4×8 DDP) | 8 (单卡) |
| iters/epoch | 3696 | 14785 |
| DINOv3 teacher | **全程启用**（epoch 12 resume 后 teacher 重新加载，未丢失） | **启用**（全程运行，console log 确认） |
| s/it | 3.5（epoch 0-11, 较慢 GPU）→ 0.5（epoch 12+, 疑似硬件升级） | 0.47-0.49（RTX 5070 Ti） |
| 蒸馏权重 | 自适应 5.0→123.77（epoch 0-11）→ resume 重置 5.0→128.55（epoch 12+） | 自适应 5.0→20.0（reset_to_default_zero_grad 触发） |
| 单 epoch 耗时 | 3.6h → 0.5h | ~2.1h |
| 有效样本/epoch | 118K | 118K（相同） |
| 梯度累积 | 无（4卡同步） | 无（小 batch） |
| AP 趋势 | 对齐 | 对齐 |

本地训练的实质：**有 DINOv3 teacher + 小 batch 的 v4-S**。RTX 5070 Ti 单卡即可在 teacher 启用的情况下达到 ~0.5 s/it（官方 4 卡 DDP 含 teacher 时 ~3.5 s/it），说明 teacher 在现代化 GPU 上不再构成瓶颈。AP 曲线与官方高度吻合证明了复现的有效性。

---

## 2026-05-22 严格验证结果（结合本文猜想 + 本地/官方日志）

本节目标是“可证据化”的结论：尽量只基于代码/配置/日志做判断，不靠主观猜想。

### 1) 本地训练是否严格贴合官方（batch-size 除外）？

结论：**贴合度很高**。除 batch-size 相关项外，本地训练与官方 `configs/rtv4/rtv4_hgnetv2_s_coco.yml` 的核心训练策略一致，且 epoch 0-14 的 AP 走势与官方非常接近。

已验证一致（证据：本地 `outputs/rtv4_hgnetv2_s_coco/console_train.log` 的 cfg 打印 + 配置文件内容）：
- 总 epoch：132
- 优化器：AdamW（主 lr=0.0004；backbone lr=0.0002；weight_decay=1e-4；norm/bn weight_decay=0）
- AMP：开启（`use_amp=True`）
- EMA：开启（`use_ema=True`，decay=0.9999）
- 梯度裁剪：`clip_max_norm=0.1`
- 蒸馏 teacher：`DINOv3TeacherModel`，权重 `pretrain/dinov3_vitb16_pretrain_lvd1689m.pth`；本地日志有明确输出 `Successfully loaded and configured DINOv3 Teacher Model.`
- 增强调度：`policy.epoch=[4,64,120]` 与 `mixup_epochs=[4,64]` 均存在（epoch4 是增强策略切换点）

已验证差异（除 batch 外仍应记录清楚）：
- 本地为了可跑通，将 `train_dataloader.total_batch_size` 从官方 32 降到 8（val 从 64 降到 16），导致 `iters/epoch` 从 3696 变为 14785 左右。
- 本地通过统一控制台覆盖了 `checkpoint_freq=1`（官方默认是 4），这只影响落盘频率，不影响训练数值。
- 本地为单卡/非分布式（官方推测为 DDP + SyncBN）。单卡下 `sync_bn=True` 通常会被跳过或退化，不等价于多卡 SyncBN，但目前 AP 对齐说明影响不大。

AP 严格对比（证据：统一控制台“官方对比表”，epoch 0-14 共 15 个点）：
- 本地-官方 AP 差值的均值约 **+0.0027**
- 最大偏差约 **+0.0149**，最小偏差约 **-0.0168**
这属于“同一量级且趋势一致”的复现结果（注意：batch 不一致时不能宣称严格复现，但可以宣称训练链路与指标趋势对齐）。

### 2) 本文关于“DINOv3 teacher 开销”的说法如何更严谨？

如果用日志来约束结论，那么“teacher 每 batch 额外 ~3 秒、约占 85%”的说法**不成立**：
- 官方日志 epoch 12+ 的训练步耗时为 **0.47~0.53 s/it**，同时 `loss_distill` 非 0 且每轮都有 `avg encoder grad | distill ...` 的蒸馏自适应调节输出，说明 teacher 在跑；因此“额外 3 秒”不可能成立。
- 本地单卡启用 teacher 时同样是 **约 0.52~0.54 s/it**（见 `console_train.log` 的 `Total time: (0.52~0.54 s/it)`），进一步说明 teacher 并非主导瓶颈。

更严谨的表述应是：teacher 会增加前向开销，但在现代 GPU 上吞吐的主导因素往往是学生网络/decoder 与整体训练图，而非 teacher 把 step 时间推到“多秒级”。

### 3) 官方 epoch 12 之后训练时长提速原因（严格验证）

现象（证据：`logs/RTv4-S-hgnet.log` 的 `Epoch: [e] Total time:`）：
- epoch 0-11：约 **3:36~3:40 / epoch**，对应 **~3.50~3.57 s/it**
- epoch 12+：约 **0:29~0:33 / epoch**，对应 **~0.47~0.53 s/it**
提速倍数约 **6.7x**（从 ~3.53 s/it 下降到 ~0.52 s/it）。

关键证据 1：蒸馏自适应权重在 epoch 12 发生“重置”
- epoch 3 后 distill 权重达到 **123.771148**，epoch 4-11 一直 `(unchanged)`
- epoch 12 行显示：`distill 5.000000 -> 8.892275`，说明 distill 权重从初始值 **5.0** 重新开始自适应爬升

关键证据 2：本地 checkpoint 的 `criterion` state_dict 为空
- 本地 `outputs/.../last.pth` 的 `criterion` 字段是空的 `OrderedDict`，说明类似 “distill 自适应权重” 这种运行时状态并不会自然保存进 checkpoint（除非项目额外实现显式保存）。
- 因此，一旦发生 **进程重启/从 checkpoint 恢复**，distill 权重从 5.0 重新开始是符合机制的。

由此可得的严格结论：
- **epoch 12 的 distill 权重重置强烈暗示训练进程在该处发生过重启/恢复**（不是简单的“训练自然推进”）。
- 但“提速的根因”并不能由 distill 权重重置解释（distill 权重是标量调节，不会带来 6-7 倍吞吐变化）。
- 因此，提速更可能来自 **重启后运行环境发生变化**（例如硬件更换、CUDA/驱动/算子实现不同、是否开启/关闭某些 debug 同步开关等）。仅凭该日志无法精确定位是哪一种，但可以确定“不是增强调度在 epoch 12 切换导致”，也不是“teacher 被关闭导致”。

建议写进论文/报告的表述（可直接引用）：
> 官方日志在 epoch 12 出现明显吞吐突变，并伴随蒸馏自适应权重的重置现象，推测该处发生了训练恢复或环境切换。由于吞吐变化幅度远超单一 loss 权重调节可解释范围，故更可能由硬件/运行环境差异引起。我们本地复现保持单一环境稳定训练，吞吐保持稳定在 ~0.52 s/it（batch=8 的 iters/epoch 更高导致 epoch 总耗时更长）。
