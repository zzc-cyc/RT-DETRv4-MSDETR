# RT-DETRv4 复现与统一控制台

> 当前阶段：只做 RT-DETRv4；v1 作为可信环境证据保留，v2 暂不纳入主线

## 已完成

- 根目录新增 `console/` 统一入口。
- 已支持 v1 / v4 双 profile。
- 已完成训练页、检测页、版本切换骨架。
- 训练页新增“官方对比”区域：可读取 v4 官方训练日志（默认 `logs/RTv4-S-hgnet.log`），并将官方 AP(best_stat)/avg_loss/epoch耗时与本地 `log.txt/console_train.log` 做逐 epoch 映射对照。
- 已补 v4 预检：`dinov3/`、`pretrain/dinov3_vitb16_pretrain_lvd1689m.pth`、COCO 路径、配置和脚本检查。
- 已通过 `python -m unittest console.test_rtdetr_console -v`。
- 已确认 `DINOv3/dinov3-main/` 已下载到本地。
- 已检查 `DINOv3/dinov3-demo/dinov3-vits16-pretrain-lvd1689m/`：这是 ViT-S/16 的 Hugging Face `model.safetensors` 包，不匹配 v4 默认需要的 ViT-B/16 `.pth` teacher 权重。
- 已确认 `DINOv3/dinov3-main/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` 已存在，是 v4 默认 teacher 所需的 ViT-B/16 LVD-1689M 权重。
- 已用 `rtdetr_5070ti` 环境试跑 `torch.hub.load`，当前阻塞点不是权重，而是缺少 DINOv3 依赖：至少 `torchmetrics`、`omegaconf`。
- 已将 `RT-DETRv4-main/dinov3` 建为指向 `DINOv3/dinov3-main` 的 junction，保持 v4 默认 `dinov3/` 配置可用。
- 已创建 `RT-DETRv4-main/pretrain/`，并复制 teacher 权重为 `pretrain/dinov3_vitb16_pretrain_lvd1689m.pth`。
- 已在 `rtdetr_5070ti` 安装 v4/DINOv3 缺失依赖：`faster-coco-eval`、`tensorboard`、`calflops`、`torchmetrics`、`omegaconf`、`ftfy`、`scikit-learn`、`submitit`、`termcolor`、`opencv-python==4.10.0.84`。
- 已将 NumPy 固定回 `1.26.4`，避免 `opencv-python` 最新版拉升到 NumPy 2.x 破坏既有训练环境。
- 已验证 DINOv3 teacher 可加载：`torch.hub.load(..., "dinov3_vitb16", source="local")` 成功，`embed_dim=768`。
- 已验证 RT-DETRv4 S checkpoint 可读取：`checkpoint/RTv4-S-hgnet.pth` 包含 `model`、`ema`、`optimizer` 等字段。
- 已通过 v4 官方 S 权重 COCO val baseline：`AP=0.498`，`AP50=0.671`，`AP75=0.540`，最大显存约 `6225 MB`。
- 已补齐官方 HGNetv2-B0 stage1 预训练权重：`pretrain/hgnetv2/PPHGNetV2_B0_stage1.pth`。
- 已修正统一控制台 v4 配置覆盖方式：所有 `-u` 覆盖项合并在同一个 `-u` 参数后，避免 argparse 只保留最后一组。
- 已启动统一控制台：`http://127.0.0.1:7861/`，当前控制台服务 PID `47524`。
- 已用官方参数启动 v4 S 从头训练链路：`seed=0`、`--use-amp`、训练 `total_batch_size=32`、验证 `total_batch_size=64`、num workers `4/4`。
- 已确认训练链路可进入正式阶段：本地 HGNetv2 与 DINOv3 teacher 加载成功，COCO train/val dataloader 构建成功，FLOPs 统计完成，GPU 显存峰值约 `11.86 GB`。
- 因电脑负载较高，已按用户要求停止训练进程 `PID=57900`；停止后显存回落到约 `0.75 GB`。

## 进行中

- v4 环境与官方参数训练链路已验证通过，当前训练已停止，控制台继续可用。
- 下一步在用户允许电脑长时间满载时，从控制台按官方参数重新启动训练并观察 epoch 0。

## 待完成

- 重新启动 v4 S 官方参数训练，并跑完至少 epoch 0。
- 记录 epoch 0 loss/AP、显存、耗时、checkpoint 和日志路径。
- 若官方参数在 12GB 显存上后续 OOM，再记录为硬件约束后决定是否采用等效 batch/梯度累积/降低 worker 等调整。

## 控制台现状

- 训练命令已按 v4 官方入口拼接。
- Resume 命令已单独处理为 `last.pth` 优先。
- 检测命令统一走 `tools/inference/torch_inf.py`。
- 页面默认只显示 v1 / v4，不再引入 v2 主线。

## 建议下一步

1. 等电脑空闲时，从控制台重新启动 v4 S 官方参数训练。
2. 先跑完 epoch 0，核对 loss/AP 是否合理。
3. 再整理 MrDETR 融合前的 baseline 记录表。

## 2026-05-20 v4 baseline 验证命令

```powershell
conda run -n rtdetr_5070ti python train.py `
  -c configs/rtv4/rtv4_hgnetv2_s_coco.yml `
  -r checkpoint/RTv4-S-hgnet.pth `
  --test-only -d cuda `
  -u val_dataloader.dataset.img_folder="D:/研究生/小论文/RT-DETR/RT-DETR-main/rtdetr_pytorch/dataset/coco/val2017" `
     val_dataloader.dataset.ann_file="D:/研究生/小论文/RT-DETR/RT-DETR-main/rtdetr_pytorch/dataset/coco/annotations/instances_val2017.json" `
     val_dataloader.num_workers=2
```

验证结果：COCO val `AP=0.498`，`AP50=0.671`，`AP75=0.540`。

## 2026-05-20 控制台白屏修复

### 已完成

- 修复统一控制台白屏：`renderCommands()` 在 `state` 未返回前先清空命令预览，避免 JS 初始化中断。
- 修复版本切换残留路径：切换 `v1/v4` 时重置版本相关表单字段，再重新生成命令。
- 补了 inline favicon，避免浏览器自动请求 `favicon.ico` 产生 404。
- 已用 `playwright-cli` 真实浏览器验证：页面可正常显示，版本下拉能看到 `RT-DETR v1 / RT-DETR v4`，训练/检测切换可用，切到 v1 后命令会切回 v1。

### 进行中

- 统一控制台继续作为训练/检测入口。
- 当前控制台服务 PID：`67840`。

### 待完成

- 电脑空闲后重新启动 v4 S 官方参数训练，并跑完至少 epoch 0。
- 记录 epoch 0 loss/AP、显存、耗时、checkpoint 和日志路径。
- 后续再把 MrDETR 融合进 v4 时，再在控制台补实验分支与开关。

## 2026-05-20 日志清空与刷新功能

### 已完成

- 训练页的“错误 / 警告日志”和“控制台日志”已新增 `清空显示`、`刷新日志` 按钮。
- 检测页的错误日志和控制台日志同样已接入 `清空显示`、`刷新日志` 按钮。
- `清空显示` 只清空当前网页显示层，不删除、不截断磁盘日志；`刷新日志` 会重新读取当前状态接口返回的最新日志。
- 已用 `playwright-cli` 真实浏览器验证：点击清空后日志区显示“已清空当前页面显示”；点击刷新后日志内容恢复。
- 已通过控制台单测：`conda run -n rtdetr_5070ti python -m unittest console.test_rtdetr_console -v`，共 16 个测试通过。

### 进行中

- 统一控制台继续运行在 `http://127.0.0.1:7861/`，当前服务 PID `67840`。
- v4 训练当前未运行，最近一次启动训练进程 `PID=65492` 已退出；日志停在 FLOPs 统计后，无 traceback/OOM 明文，后续正式训练前建议继续观察一次。

### 待完成

- 电脑空闲后，严格按官方参数重新启动 v4 S 训练，并优先确认是否进入 `Epoch: [0]` 训练循环。
- 若再次在 FLOPs 后无异常退出，继续排查 dataloader worker、Windows 子进程退出状态和日志编码输出。

## 2026-05-20 batch 8 短训链路检查

### 已完成

- 用户将 v4 S 训练参数临时调整为训练 `total_batch_size=8`、验证 `total_batch_size=16` 后启动训练。
- 训练已成功越过 FLOPs 统计并进入正式训练循环，日志出现 `Start training` 与 `Epoch: [0]`。
- 训练进度已至少到 `Epoch: [0] [4900/14785]`，说明 HGNetv2、DINOv3 teacher、COCO dataloader、AMP、loss 计算和反向传播链路均可工作。
- 错误/警告日志仅有 `GradScaler` FutureWarning 和 tensor 转 scalar UserWarning，未见 traceback、CUDA OOM 或 NaN loss。
- 该 run 已跑完 epoch 0 并完成 COCO 验证，生成 `last.pth`、`best_stg1.pth`、`log.txt`；随后在 epoch 1 刚开始时中断（`log.txt` 仅记录到 epoch 0）。

### 进行中

- batch 8 run 仅作为硬件可运行性与链路健康验证，不作为严格官方参数复现结果。

### 待完成

- 若继续严格复现，需要回到官方 `total_batch_size=32/64`；若硬件限制无法稳定启动，应单独记录为“硬件约束下 batch 8 复现实验”。
- 后续若保留 batch 8 路线，需要明确记录有效 batch、学习率、训练步数变化，并评估是否需要梯度累积来接近官方 total batch。

## 2026-05-21 RT-DETRv4-S epoch 0 对比官方（本地 batch 8/16）

### 本地（`outputs/rtv4_hgnetv2_s_coco/`）

- 本次 run 的 dataloader：训练 `total_batch_size=8`，验证 `total_batch_size=16`（来自 `console_train.log` 的 cfg 打印）。
- iters/epoch：`14785`
- epoch 0 耗时：`2:01:34`（`0.4934 s/it`）
- epoch 0 平均 loss：`30.6389`（与 `log.txt: train_loss` 一致）
- epoch 0 COCO bbox：
  - `AP=0.2553`
  - `AP50=0.3638`
  - `AP75=0.2733`
  - `APs=0.1308` / `APm=0.2859` / `APl=0.3645`

### 官方（`logs/RTv4-S-hgnet.log`）

- 官方默认 batch：`total_batch_size=32/64`（见 `configs/base/dataloader.yml`）
- iters/epoch：`3696`
- epoch 0 耗时：`3:37:58`（`3.5385 s/it`）
- epoch 0 平均 loss：`32.8500`（日志末行括号内 avg）
- epoch 0 COCO bbox：
  - `AP=0.2486`
  - `AP50=0.362`
  - `AP75=0.268`
  - `APs=0.121` / `APm=0.279` / `APl=0.354`

### 结论（你该怎么解读）

- 指标非常接近：本地 `AP=0.2553` vs 官方 `0.2486`，差约 `+0.0067`，属于“对齐到同一量级并略高”的情况。
- 但这不是严格复现结论：你的 run 是 batch 8/16，因此 iters/epoch 是官方的约 4 倍，优化轨迹与学习率-有效 batch 的关系都不同。
- 这个结果最重要的意义是：从“能不能正确训练/验证”这件事上，你的 v4 baseline 链路已经健康了。下一步只需要在硬件允许时回到官方 `32/64`，再做严格对比就可以把“严格复现”这一条证据补齐。

## 2026-05-21 RT-DETRv4-S epoch 0-4 对比官方（本地 batch 8/16）

### 核心结论（先看这个）

- 你本地 epoch 0-4 的 AP 整体在官方附近上下波动：epoch 2-3 略高，epoch 1/4 略低。
- epoch 4 是数据增强策略的切换点（开启 Mosaic 与 MixUp），本地与官方都出现训练 loss 波动；你本地在 epoch 4 有小幅回落是“合理且常见”的现象，建议继续跑 2-3 个 epoch 看是否回升并稳定。

### 指标对比（AP，官方取 best_stat）

| epoch | 本地 AP | 官方 AP | 差值(本地-官方) |
|---:|---:|---:|---:|
| 0 | 0.2553 | 0.2486 | +0.0067 |
| 1 | 0.3257 | 0.3425 | -0.0168 |
| 2 | 0.3688 | 0.3631 | +0.0057 |
| 3 | 0.3902 | 0.3753 | +0.0149 |
| 4 | 0.3726 | 0.3798 | -0.0073 |

### 训练 loss 与耗时（epoch 0-4）

- 本地 `train_loss`（来自 `log.txt`）：30.64 / 30.15 / 28.79 / 28.03 / 29.40  
- 官方 `avg loss`（来自每个 epoch 最后一个 iter 的括号内 avg）：32.85 / 27.51 / 27.40 / 28.60 / 32.39  
- 本地耗时（来自 `console_train.log`）：`2:01:34` / `2:03:32` / `2:05:49` / `2:04:31` / `2:13:08`  
- 官方耗时（来自 `logs/RTv4-S-hgnet.log`）：约 `3:37` 每个 epoch  

说明：
- loss 口径可以对比“趋势”，但不宜用绝对值下结论（batch/增强/有效学习率都在影响它）。
- 耗时不可横向比较，只能说明“你这台机器在 batch 8/16 下每 epoch 大约 2 小时量级”。

### 为什么 epoch 4 会“看起来变差/波动”

- **MixUp 从 epoch 4 才开始**：配置为 `mixup_epochs=[4,64]`，代码逻辑是 `4 <= epoch < 64` 才会 apply。  
- **Mosaic 等增强从 epoch 4 开始进入第二阶段**：`policy.epoch=[4,64,120]` 且 name 为 `stop_epoch`，epoch<4 会跳过 Mosaic/ZoomOut/IoUCrop/PhotometricDistort（NoAug），epoch>=4 才开始启用。  

这意味着 epoch 4 训练分布会明显变难，训练 loss 往往会上升或抖动，但这不一定代表泛化变差；更可靠的判断是看 epoch 5-8 的 AP 是否回到上升轨迹。

## 2026-05-21 v4 checkpoint0003 而不是 checkpoint0004 的原因与修复

### 原因

- v4 配置里 `checkpoint_freq=4`（见 `configs/base/dfine_hgnetv2.yml`），保存逻辑在 `engine/solver/det_solver.py`：
  - 只有当 `(epoch + 1) % checkpoint_freq == 0` 才会额外保存 `checkpoint000X.pth`
  - 因此 epoch 0-4 阶段只会在 epoch 3 结束后产生 `checkpoint0003.pth`，epoch 4 结束不会产生 `checkpoint0004.pth`
- 同时 `last.pth` 会在每轮更新，所以你会看到 `last.pth` 很新，但 `checkpoint000X.pth` 不连续。

### 修改（每轮都保存）

- 统一控制台已为 v4 训练命令增加 `checkpoint_freq=1` 覆盖项：每个 epoch 结束都会保存一次 `checkpoint000X.pth`。
- 注意磁盘占用：单个 v4 checkpoint 约 170MB，长跑会快速增长。

## 2026-05-21 官方训练日志对比映射修复（epoch>=5 不再显示 `-`）

### 问题现象

- 统一控制台“官方对比”表中，某些 epoch 的“官方 AP”显示为 `-`，尤其在 epoch 5+ 更容易遇到。

### 根因

- v4 官方日志中的 `best_stat` 不是“每个 epoch 都打印”，它只会在出现新 best 时更新。
- 因此如果只用 `best_stat.coco_eval_bbox` 作为“官方 AP”，当该 epoch 没有 best_stat 就会缺失。

### 修复方案（已实现）

- 控制台对官方 AP 的读取逻辑改为：
  - 优先使用 `best_stat`（表格标注为 `best`）。
  - 若该 epoch 没有 best_stat，则回退到该 epoch 的评估块 AP（表格标注为 `epoch`）。
- UI 的“官方 AP”单元格会显示来源标注（`best` / `epoch`），避免误解为“没读取到官方日志”。

### 验证

- 已新增单测覆盖 fallback：`test_official_compare_falls_back_to_epoch_ap_when_best_stat_missing`。
- `/api/status` 返回对比数据中包含 `official_ap_source`（`best_stat` / `epoch_ap` / `missing`）。

## 2026-05-22 本地与官方训练“严格贴合度”验证（batch-size 除外）

### 已完成

- 已结合 `RT-DETRv4-S 官方训练方法完整解析.md` 的猜想、官方日志 `logs/RTv4-S-hgnet.log` 与本地训练日志 `outputs/rtv4_hgnetv2_s_coco/console_train.log` 做严格验证。
- 结论：本地训练除 `total_batch_size`（8/16 vs 官方 32/64）外，核心训练配置与官方一致（epoch=132、flatcosine、AdamW 参数组、AMP/EMA、teacher 配置与蒸馏自适应参数、epoch4 增强调度切换点均一致），且 epoch 0-14 的 AP 差值均在小范围内波动（均值约 +0.0027，最大 +0.0149，最小 -0.0168）。

### 关键发现：官方 epoch 12 之后“提速”原因

- 官方日志在 epoch 12 出现 `s/it` 从 ~3.53 降到 ~0.52 的突变，同时蒸馏自适应权重从 `123.771148` 重置回初始值 `5.0` 并重新爬升。
- 这强烈暗示 epoch 12 处发生过“进程重启/Resume”（distill 自适应权重属于运行时状态，默认不会随 checkpoint 保存；本地 `last.pth` 的 `criterion` state_dict 为空即可证明这类状态不随 checkpoint 持久化）。
- 但 distill 权重重置无法解释 6-7 倍吞吐提升，因此更可能是重启后环境/硬件发生变化（仅凭日志无法严格定位具体是哪种变化）。

### 进行中

- v4 训练继续运行（本地吞吐稳定在 ~0.52 s/it；epoch 总耗时约 2 小时是因为 `iters/epoch`=14785，源于 batch=8）。

### 待完成

- 若目标升级为“严格官方参数复现”，需要恢复到官方 `total_batch_size=32/64` 或引入梯度累积实现等效 total batch，并单独记录为新的实验 run（避免与 batch=8 的曲线混淆）。
- 2026-05-22 决策：本轮训练先不引入梯度累积；仅当本轮复现结果不理想或出现系统性偏差时，再实现并打开控制台开关进行对照实验。

## 2026-05-22 v4-S baseline 是否继续跑满 132 epoch 的阶段决策

### 已完成

- 当前本地 v4-S 在 epoch 0-15 的 loss/AP 与官方训练日志基本持平，已足以证明训练链路、DINOv3 teacher、损失项、增强调度和评估流程健康。
- 官方 checkpoint 的 COCO val baseline 已验证过（AP=0.498），可作为最终官方性能锚点；本地从头训练的前 15 个 epoch 可作为“复现趋势证据”。

### 进行中

- 优先从“完整跑满 132 epoch”切换为“固化当前 baseline 证据 + 启动模块融合小实验”。

### 待完成

- 建议在当前 epoch 完成验证并保存 checkpoint 后停止，不建议中断在 epoch 中间。
- 固化当前 run 的 `log.txt`、`console_train.log`、`last.pth/checkpointXXXX.pth`、控制台官方对比截图或导出数据。
- 后续融合 MrDETR/MSDETR 模块时，先做 1-3 epoch sanity check，再做 15-20 epoch 与当前 baseline 同条件对比；只有效果明显有希望时再扩展到 40/64/132 epoch。
