# 项目协作规则

用户是研一学生，目标是复现 RT-DETR 论文实验结果。现在开始你就是我的ai导师，要根据实际情况给我提出合适的建议，指导我完成复现并有所创新。
我想复现全部的RT-DETR系列（最新已经到v4），然后将MRDETR的模块融合进v4。所有源码和论文已经在当前文件夹下。我目前的进度是复现到了RT-DETRv1版本的r18模型，虽然只跑了六轮epoch，但是效果和论文接近，所以理论上可以完整复现。希望可以将整个项目整理，然后给出我一个可行性报告和实现流程指南。

官方项目链接：https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr\_pytorch
https://github.com/RT-DETRS/RT-DETRV4?tab=readme-ov-file#4-citation
https://github.com/Visual-AI/Mr.DETR



## 操作约束

* 用中文回答。
* 不清楚或不确定的问题需要先向用户询问后对齐。
* 禁止批量删除文件或目录。
* 不要使用 `del/s`、`rd/s`、`rmdir /s`、`Remove-Item -Recurse`、`rm -rf`。
* 需要删除文件时，只能一次删除一个明确路径的文件，例如 `Remove-Item "C:\\path\\to\\file.txt"`。
* 每次执行完任务或一个阶段完成后，使用 `neat-freak` skill 做收尾整理。
* 每次阶段更新都要维护 Markdown 文档，并列出“已完成、进行中、待完成”。

## 当前环境

* 项目根目录：`D:\\研究生\\代码复现\\RT-DETR-main`
* 复现入口：`rtdetr\_pytorch`
* 虚拟环境：`rtdetr\_5070ti`
* 实际环境：Python 3.11.14，PyTorch 2.8.0+cu129，torchvision 0.23.0+cu129，CUDA 12.9
* 数据集：`rtdetr\_pytorch/dataset/coco`
* 优先权重：`weights/rtdetr\_r18vd\_dec3\_6x\_coco\_from\_paddle.pth`

## 复现进度

### 已完成

* 已梳理本地项目结构，确认包含 `rtdetr\_pytorch`、`rtdetr\_paddle`、`rtdetrv2\_pytorch`、`rtdetrv2\_paddle`。
* 已严格阅读官方 RT-DETR 根 README 与 `rtdetr\_pytorch/README.md`。
* 已确认本地 COCO 2017 数据完整：train2017 为 118287 张，val2017 为 5000 张。
* 已确认本地存在官方 R18 COCO 权重和 R18 训练日志。
* 已创建 `复现指南.md`，聚焦 RT-DETR-R18 评估和训练复现。
* 已增加 torchvision 0.23 兼容层：`rtdetr\_pytorch/src/data/datapoints\_compat.py`，并更新 `rtdetr\_pytorch/src/data/transforms.py`。
* 已完成官方 R18 COCO 权重评估：AP 0.464，AP50 0.637，AP75 0.503，对齐官方 PyTorch README。
* 已新增图形化训练控制台：`rtdetr\_pytorch/tools/training\_dashboard.py`。
* 已新增控制台测试：`rtdetr\_pytorch/tests/test\_training\_dashboard.py`。
* 已在 `http://127.0.0.1:7861/` 验证当前仓库控制台页面和 `/api/status`；预检全绿。
* 已给控制台启动参数区增加“当前启动训练指令”和“当前 Resume 指令”预览，并验证会随 Seed、AMP 等参数实时更新。
* 已定位并修复训练启动退出问题：`SanitizeBoundingBox` 旧配置名未注册，以及 `datapoints.BoundingBox` 兼容层不能作为类型参与 torchvision v2 检查。
* 已新增 transform 兼容测试：`rtdetr\_pytorch/tests/test\_transforms\_compat.py`。
* 已于 2026-05-18 15:21 启动 RT-DETR-R18 72 epochs 正式训练，控制台 PID 为 `109236`，启动命令为 `python tools/train.py -c configs/rtdetr/rtdetr\_r18vd\_6x\_coco.yml --seed 42`。
* 已发现该 run 的 epoch 0 异常：`train\_loss` 约 45261，`AP=0.0`，原因是 `ConvertBox` 未处理 torchvision 0.23 `BoundingBoxes` 基类，导致 boxes 未归一化。
* 已停止异常 run，并将其产物归档到 `rtdetr\_pytorch/output/invalid\_runs/r18\_bbox\_not\_normalized\_2026-05-18/`。
* 已修复 `ConvertBox` 兼容范围，并验证真实 COCO 训练 batch 的 boxes 已归一化到 0-1。
* 已完成严格复查：补齐 `PadToSize` 对 torchvision 0.23 `BoundingBoxes` 的兼容；控制台从头训练改为覆盖 `console\_train.log` 和 `console\_train.err.log`，Resume 才追加日志。
* 已重启 `http://127.0.0.1:7861/` 当前仓库控制台，输出目录当前无正式训练 `log.txt/checkpoint.pth`，预检全绿。

### 进行中

* 准备重新启动 RT-DETR-R18 72 epochs 完整训练，并重点核对 epoch 0 的 loss/AP 是否回到官方量级。

### 待完成

* 重新启动修复后的正式训练。
* 核对修复后 epoch 0：`train\_loss` 应接近 15.46，AP 应接近 0.1605。
* 使用训练控制台监控 AP/loss 曲线、checkpoint、GPU 和日志。
* 对比本地训练日志与 `rtdetr\_pytorch/references\_log/rtdetr\_r18vd\_6x\_coco\_pytorch\_log.txt`。
* 记录硬件、PyTorch、CUDA、训练时长和随机种子。

## v1 复现与控制台（摘要归档）

### 已完成

* v1（rtdetr_pytorch）训练/评估链路已跑通，并完成 epoch 0-6 的对齐验证，趋势与官方参考一致，可作为“环境可信”证据。
* v1 训练控制台能力已形成并验证过（参数预览、日志、checkpoint、状态接口、兼容层与单测）。

### 进行中

* v1 的 72 epochs 完整跑完不再作为 v4 的硬前置，仅作为可选补充；主线已切到 v4 复现与后续 MrDETR 融合。

### 待完成

* 若后续需要把 v1 的完整训练作为论文补充证据，再从 v1 输出目录 Resume 继续即可（并在对应复现文档记录参数与随机种子）。

## 2026-05-19 全系列复现与融合规划

### 已完成

* 已通读 RT-DETR v1/v2、RT-DETRv4、MrDETR 关键源码路径。
* 已更新 [可行性报告与实现流程指南.md](./可行性报告与实现流程指南.md)，路线从“先完整复现 v1/v2”调整为“v1证据归档 + v2轻量参照 + v4优先复现 + MrDETR融合”。
* 已确认融合高度可行：MrDETR 多路由训练是训练策略，与 v4 的 D-FINE decoder 兼容；One2Many Matcher 本地源码 cost 为 `0.7 \* IoU + 0.3 \* class\_score`。
* 已明确 v1 R18 epoch 0-6 结果可作为环境与复现可信度证据，不再把 v1 72 epochs 和 v2 120 epochs 作为进入 v4 的硬前置。

### 进行中

* 当前工作目录为 `D:\\研究生\\小论文`，三个子项目已就位。
* 主线已切换到 RT-DETRv4：官方 S 权重 baseline val 已跑通，下一步做 1-3 epoch 短训 sanity check。

### 待完成（按Phase顺序）

* **Phase 0**：归档 v1 epoch 0-6 checkpoint/log/环境信息；72 epochs 仅作为可选补充。
* **Phase 1**：v2 轻量参照；必要时只做权重评估，不做 120 epochs 从头训练。
* **Phase 2**：RT-DETRv4 baseline 跑通：官方权重 val 评估 + 1-3 epochs 短训。
* **Phase 3**：MrDETR融合到v4（6个核心步骤，详见指南）。
* **Phase 4**：训练验证 + 消融 + baseline 对比。

## 2026-05-20 RT-DETRv4 环境验证

### 已完成

* `DINOv3/dinov3-main/` 已下载完整，v4 目录下 `dinov3` 已建为 junction 指向该目录。
* 已将 `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` 复制为 `RT-DETRv4/RT-DETRv4-main/pretrain/dinov3_vitb16_pretrain_lvd1689m.pth`。
* 已在 `rtdetr_5070ti` 环境补齐 v4/DINOv3 依赖，并将 NumPy 固定回 `1.26.4`、OpenCV 固定为 `4.10.0.84`。
* DINOv3 teacher 加载成功：`dinov3_vitb16`，`embed_dim=768`。
* 官方 checkpoint `RT-DETRv4/RT-DETRv4-main/checkpoint/RTv4-S-hgnet.pth` 已读取成功。
* 官方 S 权重 COCO val baseline 已跑通：`AP=0.498`，`AP50=0.671`，`AP75=0.540`，最大显存约 `6225 MB`。
* 统一控制台测试仍通过：`python -m unittest console.test_rtdetr_console -v`。
* 已补齐 `pretrain/hgnetv2/PPHGNetV2_B0_stage1.pth`，解决从头训练首次加载 HGNetv2 预训练权重时误入自动下载分支的问题。
* 统一控制台 v4 命令已修正为单个 `-u` 后接全部配置覆盖项，保证 COCO 路径覆盖生效。
* 已用官方训练参数启动 v4 S 从头训练链路：`seed=0`、`--use-amp`、训练 `total_batch_size=32`、验证 `total_batch_size=64`、num workers `4/4`。
* 训练已确认可进入正式链路：HGNetv2/DINOv3 加载成功，COCO train/val dataloader 构建成功，FLOPs 统计完成，显存峰值约 `11.86 GB`。
* 因用户电脑负载较高，已停止训练 PID `57900`；停止后显存约 `0.75 GB`。

## 2026-05-20 统一控制台白屏修复

### 已完成

* 修复统一控制台白屏：`renderCommands()` 在 `state` 未返回前先清空命令预览，避免页面初始化时 JS 中断。
* 修复版本切换残留路径：切换 `v1/v4` 时重置版本相关表单字段，再重新生成命令。
* 补了 inline favicon，避免浏览器自动请求 `favicon.ico` 产生 404。
* 已用 `playwright-cli` 真实浏览器验证：页面可正常显示，版本下拉能看到 `RT-DETR v1 / RT-DETR v4`，训练/检测切换可用，切到 v1 后命令会切回 v1。

### 进行中

* 统一控制台继续作为训练/检测入口，当前控制台服务 PID `97056`。

### 待完成

* 电脑空闲后重新启动 v4 S 官方参数训练，并跑完至少 epoch 0。
* 记录 epoch 0 loss/AP、显存、耗时、checkpoint 和日志路径。
* 后续再把 MrDETR 融合进 v4 时，再在控制台补实验分支与开关。

## 2026-05-20 日志清空与刷新功能

### 已完成

* 统一控制台训练页的“错误 / 警告日志”和“控制台日志”已新增 `清空显示`、`刷新日志` 按钮。
* 统一控制台检测页的错误日志和控制台日志也已接入 `清空显示`、`刷新日志` 按钮。
* `清空显示` 只清空当前网页显示层，不删除、不截断磁盘日志；`刷新日志` 会重新读取状态接口返回的最新日志。
* 已用 `playwright-cli` 真实浏览器验证清空与刷新交互：清空后显示提示文本，刷新后日志恢复。
* 已通过验证：`conda run -n rtdetr_5070ti python -m unittest console.test_rtdetr_console -v`，16 个测试通过。

### 进行中

* 控制台保持运行在 `http://127.0.0.1:7861/`，当前服务 PID `97056`。
* v4 训练当前未运行；最近一次训练进程 `PID=65492` 已退出，日志停在 FLOPs 统计后，无 traceback/OOM 明文，后续重新启动时需要重点观察是否进入 `Epoch: [0]`。

### 待完成

* 等电脑空闲后，重新启动 v4 S 官方参数训练并跑完 epoch 0。
* 记录 epoch 0 loss/AP、显存、耗时、checkpoint 和日志路径。
* 短训稳定后再进入 MrDETR 融合设计与最小实现。

## 2026-05-20 batch 8 短训链路检查

### 已完成

* 用户将 v4 S 训练参数临时调整为训练 `total_batch_size=8`、验证 `total_batch_size=16` 后启动训练。
* 训练已成功越过 FLOPs 统计并进入正式训练循环，日志出现 `Start training` 与 `Epoch: [0]`。
* 训练进度已至少到 `Epoch: [0] [4900/14785]`，说明 HGNetv2、DINOv3 teacher、COCO dataloader、AMP、loss 计算和反向传播链路均可工作。
* 错误/警告日志仅有 `GradScaler` FutureWarning 和 tensor 转 scalar UserWarning，未见 traceback、CUDA OOM 或 NaN loss。
* 该 run 已跑完 epoch 0 并完成 COCO 验证，生成 `last.pth`、`best_stg1.pth`、`log.txt`；随后在 epoch 1 刚开始时中断（`log.txt` 仅记录到 epoch 0）。

### 进行中

* batch 8 run 已完成 epoch 0，可作为“链路健康 + 指标接近官方”的证据；但由于 batch 与官方不一致，不能作为严格官方参数复现结果。
* 严格官方参数（`total_batch_size=32/64`）的从头训练仍未完成 epoch 0。

### 待完成

* 若继续严格复现，需要回到官方 `total_batch_size=32/64`；若硬件限制无法稳定启动，应单独记录为“硬件约束下 batch 8 复现实验”。
* 后续若保留 batch 8 路线，需要明确记录有效 batch、学习率、训练步数变化，并评估是否需要梯度累积来接近官方 total batch。

## 2026-05-21 RT-DETRv4-S epoch 0 与官方日志对比（本地 batch 8/16）

### 已完成

* 本地输出目录：`RT-DETRv4/RT-DETRv4-main/outputs/rtv4_hgnetv2_s_coco/`，epoch 0 已产出 `log.txt` 与 checkpoint。
* 从本地 `console_train.log` 确认本次 run 的 dataloader 设置为训练 `total_batch_size=8`、验证 `total_batch_size=16`（与官方默认 `32/64` 不一致）。
* 指标对比（epoch 0）：
  * 本地：`AP=0.2553`，`AP50=0.3638`，`AP75=0.2733`，`APs=0.1308`，`APm=0.2859`，`APl=0.3645`；epoch 0 平均 loss `30.6389`；耗时 `2:01:34`，iters/epoch=`14785`。
  * 官方（`logs/RTv4-S-hgnet.log`）：`AP=0.2486`，`AP50=0.362`，`AP75=0.268`，`APs=0.121`，`APm=0.279`，`APl=0.354`；epoch 0 平均 loss `32.8500`（按日志末行括号内 avg）；耗时 `3:37:58`，iters/epoch=`3696`。
* 结论：本地 epoch 0 AP 与官方非常接近（略高约 `+0.0067`），说明训练链路与评估链路整体正确；但由于 batch/iters 口径不同，不应把该结果记为“严格官方参数复现完成”。

### 进行中

* 计划在硬件允许时，回到官方默认 `total_batch_size=32/64` 完成 epoch 0（作为严格复现基准）。

### 待完成

* 对齐官方 batch 后再做 epoch 0/1 指标对比（以及耗时、显存峰值、checkpoint 产物一致性）。

## 2026-05-21 RT-DETRv4-S epoch 0-4 与官方日志对比（本地 batch 8/16）

### 已完成

* 本地 `outputs/rtv4_hgnetv2_s_coco/log.txt` 已记录到 epoch 4（共 5 行：epoch 0-4）。
* 本地训练/验证 batch：train `total_batch_size=8`、val `total_batch_size=16`（与官方 `32/64` 不一致，iters/epoch 为官方约 4 倍）。
* 指标对比（AP，官方取 `best_stat.coco_eval_bbox`，本地取 `log.txt:test_coco_eval_bbox[0]`）：
  * epoch 0：`+0.0067`（本地略高）
  * epoch 1：`-0.0168`（本地略低）
  * epoch 2：`+0.0057`（本地略高）
  * epoch 3：`+0.0149`（本地略高）
  * epoch 4：`-0.0073`（本地略低）
* 详细对照表已写入 `RT-DETRv4复现与统一控制台.md` 的“epoch 0-4 对比官方”小节（避免 AGENTS.md 膨胀）。
* 统一控制台训练页已新增“官方对比”区域：自动加载默认官方日志并生成对照表（支持手动改官方日志路径）。
* 已修复“官方对比未及时映射”的问题：当某个 epoch 的 `best_stat` 没有更新时（官方日志常见现象），控制台会自动回退到该 epoch 的评估块 AP（`epoch_ap`），避免表格显示 `-`。
* 官方对比表的“官方 AP”列会显示来源标注：`best`（来自 best_stat）或 `epoch`（来自该 epoch 评估块）。
* 现象解释（重要）：epoch 4 是数据增强调度的分界点（`policy.epoch=[4,64,120]`，`mixup_epochs=[4,64]`），会在 epoch>=4 开启 Mosaic/ZoomOut/IoUCrop/PhotometricDistort 与 MixUp；因此本地与官方在 epoch 4 都出现“训练 loss 上升/波动”的特征，本地 AP 也可能短暂回落。
* 训练耗时对比（epoch 0-4）：
  * 本地约 `~2:01` 到 `~2:13` / epoch（14785 iters/epoch，约 `0.49~0.54 s/it`）
  * 官方约 `~3:35` 到 `~3:38` / epoch（3696 iters/epoch，约 `3.50~3.54 s/it`）
  * 说明：耗时不可直接横比（硬件/分布式/iters 口径不同），主要看“能稳定跑、指标对齐趋势”。

### 进行中

* 继续跑到 epoch 6-8，观察 epoch 4 分界点后的 AP 是否回升并稳定贴近官方曲线。

### 待完成

* 若目标是“严格官方参数复现”，需要用官方 `total_batch_size=32/64` 或引入梯度累积实现等效 total batch，再重新对齐 epoch 0-4 曲线。

## 2026-05-21 v4 每轮保存 checkpoint（checkpoint_freq=1）

### 已完成

* 原因定位：v4 的 `checkpoint_freq` 默认是 `4`，保存逻辑为 `(epoch+1) % checkpoint_freq == 0`，所以 epoch0-4 阶段只会在 epoch3 结束后生成 `checkpoint0003.pth`，不会在 epoch4 结束时生成 `checkpoint0004.pth`。
* 统一控制台已对 v4 训练命令强制追加 `checkpoint_freq=1` 覆盖项，使其每个 epoch 结束都保存一次 `checkpoint000X.pth`。
* 已通过控制台单测：`python -m unittest console.test_rtdetr_console -v`。

### 进行中

* 下一次从控制台启动/Resume v4 训练后，观察是否在 epoch4/5 结束产生 `checkpoint0004.pth` / `checkpoint0005.pth`。

### 待完成

* 评估磁盘占用：每个 checkpoint 大约 170MB，长期训练会显著增长，需要提前规划保留策略（例如只保留关键 epoch）。

## 2026-05-22 本地 vs 官方“严格贴合度”验证（batch-size 除外）+ 官方 epoch12 提速原因

### 已完成

* 已结合 `RT-DETRv4-S 官方训练方法完整解析.md`、官方日志 `RT-DETRv4/RT-DETRv4-main/logs/RTv4-S-hgnet.log` 与本地日志 `outputs/rtv4_hgnetv2_s_coco/console_train.log` 做证据化验证。
* 验证结论：本地除 `total_batch_size=8/16`（官方 32/64）外，核心训练配置与官方配置文件保持一致（epoch=132、flatcosine、AdamW 参数组、AMP/EMA、teacher 与蒸馏自适应参数、epoch4 增强切换点等）。
* 指标结论：epoch 0-14 的 AP 与官方非常接近（差值均值约 +0.0027，最大 +0.0149，最小 -0.0168）。
* 官方 epoch12 提速原因（严格证据链）：
  * 官方日志在 epoch12 出现吞吐突变：`s/it` 从 ~3.53 降到 ~0.52。
  * 同时 `distill` 自适应权重从 `123.771148` 重置为 `5.0` 并重新爬升，强烈暗示 epoch12 处发生过“进程重启/Resume”。
  * 本地 `last.pth` 中 `criterion` state_dict 为空，证明类似 distill 自适应权重属于运行时状态，默认不会随 checkpoint 保存；因此 Resume 后重置是机制一致的。
  * 但 distill 权重重置无法解释 6-7 倍吞吐提升，因此更可能是重启后环境/硬件发生变化（仅凭日志无法严格定位具体原因）。

### 进行中

* v4 训练仍在继续（本地吞吐 ~0.52 s/it 稳定；epoch 总耗时约 2 小时主要由 iters/epoch=14785 导致）。

### 待完成

* 若要升级为“严格官方参数复现”，需要回到官方 `total_batch_size=32/64` 或引入梯度累积实现等效 total batch，并单独记录为新的实验 run。
* 2026-05-22 决策：**本轮训练先不做梯度累积改造**，优先把当前 run 跑出可对照的完整曲线；若最终复现不理想，再启用“等效 batch=32 的梯度累积”作为后续改进方案。

## 2026-05-22 v4-S baseline 阶段决策：不强行跑满 132 epoch

### 已完成

* 当前本地 v4-S epoch 0-15 的 loss/AP 与官方训练日志基本持平，足以作为“训练链路复现成功”的阶段证据。
* 官方 checkpoint val baseline 已完成，可作为最终官方性能锚点；本地从头训练前 15 epoch 用于证明趋势与训练机制对齐。

### 进行中

* 建议在当前 epoch 完成验证并保存 checkpoint 后停止 baseline 长跑，转入 MrDETR/MSDETR 模块融合的小实验。

### 待完成

* 停止前固化当前 run 的 `log.txt`、`console_train.log`、`last.pth/checkpointXXXX.pth` 与官方对比表。
* 融合实验先做 1-3 epoch sanity check，再做 15-20 epoch 同条件对比；只有出现明显收益趋势时，再扩展到更长训练。

## 备注：目录结构与融合路线

与“目录结构/融合 6 步”相关的长期信息已收敛到 `可行性报告与实现流程指南.md` 与 `RT-DETRv4复现与统一控制台.md`，避免 AGENTS.md 继续膨胀。

## 2026-05-22 Git 工程初始化

### 已完成

* 已在 `D:\研究生\小论文` 初始化 Git 仓库。
* 已新增根目录 `.gitignore`，排除 COCO/下载数据集、权重、预训练文件、checkpoint、评估缓存、压缩包、Python 缓存、浏览器/Agent 临时文件等。
* 已按用户要求保留训练日志跟踪能力：`*.log` 不再作为通用忽略项，关键 `log.txt`、`console_train.log`、官方参考日志可纳入 Git 作为复现实验证据。
* 已修正 `.gitignore` 对源码型 `dataset/datasets` 目录的误伤风险，DINOv3/MrDETR/v4 的 dataset 模块不会被排除。
* 已排除 `RT-DETRv4/RT-DETRv4-main/dinov3/` junction，避免重复跟踪同一份 DINOv3 源码；真实源码保留在 `DINOv3/dinov3-main/`。
* 已新增 `.gitattributes`，固定源码/文档换行策略，减少 Windows/Linux 换行噪音。
* 已新增 `Git工程管理.md`，记录 Git 跟踪策略、排除策略和常用检查命令。

### 进行中

* GitHub 仓库已建立远程跟踪：`origin/main`。
* 本机未安装 GitHub CLI `gh`，本次使用原生 Git 命令发布；后续若要创建 PR 或管理 Issue，建议安装并登录 `gh`。

### 待完成

* 后续进入 MrDETR/MSDETR 融合实验前，基于 `main` 新建实验分支。

## 2026-05-22 GitHub 发布记录

### 已完成

* 已创建初始化提交：`a51e409`，提交信息为 `Initialize RT-DETRv4 MSDETR workspace`。
* 已添加远程仓库：`origin = https://github.com/zzc-cyc/RT-DETRv4-MSDETR.git`。
* 已推送 `main` 到 `origin/main`，并建立本地分支跟踪关系。
* 推送前已验证暂存区无 `10MB` 以上文件，且无权重、COCO 数据集、预训练、checkpoint、压缩包和 `dinov3` junction 重复路径。

### 进行中

* 当前 `main` 分支作为项目初始基线。

### 待完成

* 后续新增融合代码时，建议从 `main` 新建功能分支，例如 `msdetr/fusion-prototype`。
