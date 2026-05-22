# 项目协作规则

用户是研一学生，目标是复现 RT-DETR 论文实验结果。优先复现官方 PyTorch 版 RT-DETR-R18。

官方项目链接：https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch

## 操作约束

- 用中文回答。
- 不清楚或不确定的问题需要先向用户询问后对齐。
- 禁止批量删除文件或目录。
- 不要使用 `del/s`、`rd/s`、`rmdir /s`、`Remove-Item -Recurse`、`rm -rf`。
- 需要删除文件时，只能一次删除一个明确路径的文件，例如 `Remove-Item "C:\path\to\file.txt"`。
- 每次执行完任务或一个阶段完成后，使用 `neat-freak` skill 做收尾整理。
- 每次阶段更新都要维护 Markdown 文档，并列出“已完成、进行中、待完成”。

## 当前环境

- 项目根目录：`D:\研究生\代码复现\RT-DETR-main`
- 复现入口：`rtdetr_pytorch`
- 虚拟环境：`rtdetr_5070ti`
- 实际环境：Python 3.11.14，PyTorch 2.8.0+cu129，torchvision 0.23.0+cu129，CUDA 12.9
- 数据集：`rtdetr_pytorch/dataset/coco`
- 优先权重：`weights/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth`

## 复现进度

### 已完成

- 已梳理本地项目结构，确认包含 `rtdetr_pytorch`、`rtdetr_paddle`、`rtdetrv2_pytorch`、`rtdetrv2_paddle`。
- 已严格阅读官方 RT-DETR 根 README 与 `rtdetr_pytorch/README.md`。
- 已确认本地 COCO 2017 数据完整：train2017 为 118287 张，val2017 为 5000 张。
- 已确认本地存在官方 R18 COCO 权重和 R18 训练日志。
- 已创建 `复现指南.md`，聚焦 RT-DETR-R18 评估和训练复现。
- 已增加 torchvision 0.23 兼容层：`rtdetr_pytorch/src/data/datapoints_compat.py`，并更新 `rtdetr_pytorch/src/data/transforms.py`。
- 已完成官方 R18 COCO 权重评估：AP 0.464，AP50 0.637，AP75 0.503，对齐官方 PyTorch README。
- 已新增图形化训练控制台：`rtdetr_pytorch/tools/training_dashboard.py`。
- 已新增控制台测试：`rtdetr_pytorch/tests/test_training_dashboard.py`。
- 已在 `http://127.0.0.1:7861/` 验证当前仓库控制台页面和 `/api/status`；预检全绿。
- 已给控制台启动参数区增加“当前启动训练指令”和“当前 Resume 指令”预览，并验证会随 Seed、AMP 等参数实时更新。
- 已定位并修复训练启动退出问题：`SanitizeBoundingBox` 旧配置名未注册，以及 `datapoints.BoundingBox` 兼容层不能作为类型参与 torchvision v2 检查。
- 已新增 transform 兼容测试：`rtdetr_pytorch/tests/test_transforms_compat.py`。
- 已于 2026-05-18 15:21 启动 RT-DETR-R18 72 epochs 正式训练，控制台 PID 为 `109236`，启动命令为 `python tools/train.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml --seed 42`。
- 已发现该 run 的 epoch 0 异常：`train_loss` 约 45261，`AP=0.0`，原因是 `ConvertBox` 未处理 torchvision 0.23 `BoundingBoxes` 基类，导致 boxes 未归一化。
- 已停止异常 run，并将其产物归档到 `rtdetr_pytorch/output/invalid_runs/r18_bbox_not_normalized_2026-05-18/`。
- 已修复 `ConvertBox` 兼容范围，并验证真实 COCO 训练 batch 的 boxes 已归一化到 0-1。
- 已完成严格复查：补齐 `PadToSize` 对 torchvision 0.23 `BoundingBoxes` 的兼容；控制台从头训练改为覆盖 `console_train.log` 和 `console_train.err.log`，Resume 才追加日志。
- 已重启 `http://127.0.0.1:7861/` 当前仓库控制台，输出目录当前无正式训练 `log.txt/checkpoint.pth`，预检全绿。

### 进行中

- 准备重新启动 RT-DETR-R18 72 epochs 完整训练，并重点核对 epoch 0 的 loss/AP 是否回到官方量级。

### 待完成

- 重新启动修复后的正式训练。
- 核对修复后 epoch 0：`train_loss` 应接近 15.46，AP 应接近 0.1605。
- 使用训练控制台监控 AP/loss 曲线、checkpoint、GPU 和日志。
- 对比本地训练日志与 `rtdetr_pytorch/references_log/rtdetr_r18vd_6x_coco_pytorch_log.txt`。
- 记录硬件、PyTorch、CUDA、训练时长和随机种子。
## 2026-05-18 当前接手状态

### 已完成
- 新增每轮 checkpoint 显式保留功能：`rtdetr_pytorch/src/solver/checkpointing.py`。
- `tools/train.py` 已支持 `--checkpoint-step` 与 `--checkpoint-name-style {official,underscore}`。
- `tools/training_dashboard.py` 已在启动参数区支持 checkpoint 保存间隔和命名风格，默认每轮保存 `checkpoint_0000.pth` 风格文件。
- 相关测试已通过：`python -m unittest tests.test_checkpointing tests.test_training_dashboard tests.test_transforms_compat -v`。
- 7861 控制台已重启，新控制台 PID 为 `128860`。
- 发现旧训练 PID `128988` 已退出后，已从 `checkpoint.pth` Resume，当前训练 PID 为 `132968`。

### 进行中
- RT-DETR-R18 正在从 epoch 0 checkpoint 继续训练，等待 epoch 1 完整评估结果。

### 待完成
- 确认 epoch 1 结束后生成 `checkpoint_0001.pth`。
- 对比 epoch 1 AP 与官方参考值约 `0.245`。
- 继续监控 72 epochs 完整训练。
## 2026-05-19 epoch 1 检查记录

### 已完成
- epoch 1 已完成验证：`train_loss=13.5942`，`AP=0.2420`，`AP50=0.3482`，`AP75=0.2603`。
- 与官方参考 epoch 1：`train_loss=13.4092`，`AP=0.2452`，`AP50=0.3548`，`AP75=0.2622` 基本对齐。
- 每轮 checkpoint 功能已验证：`rtdetr_pytorch/output/rtdetr_r18vd_6x_coco/checkpoint_0001.pth` 已生成。

### 进行中
- 当前训练 PID 为 `32700`，正在继续 epoch 2。

### 待完成
- epoch 2 完成后对比官方参考 AP 约 `0.2734`。
- 持续检查 `checkpoint_0002.pth` 是否生成，以及日志中是否有新错误。
## 2026-05-19 控制台图表更新

### 已完成
- 训练控制台 AP 曲线和 Loss/LR 曲线已增加横坐标 epoch 刻度与 `Epoch` 标签。
- 已新增测试：`TrainingDashboardTests.test_chart_html_labels_x_axis_as_epoch`。
- 已通过 `python -m unittest tests.test_training_dashboard -v`。
- 7861 控制台已重启，新控制台 PID 为 `51756`。
- 当前训练进程 PID `32700` 未被中断。

### 进行中
- 训练继续运行，控制台继续监控。

### 待完成
- epoch 2 完成后检查指标与图表横轴显示。
## 2026-05-19 epoch 0-6 阶段状态

### 已完成
- 本地 epoch 0-6 已完成，并与官方 `references_log/rtdetr_r18vd_6x_coco_pytorch_log.txt` 逐轮对比。
- epoch 6 本地结果：`train_loss=11.6962`，`AP=0.3542`，`AP50=0.4987`，`AP75=0.3814`。
- 官方 epoch 6 参考：`train_loss=11.5798`，`AP=0.3505`，`AP50=0.4917`。
- 判断：复现趋势正常，epoch 2-6 AP 略高于官方参考，整体非常理想。
- 已确认 `checkpoint_0006.pth` 与最新 `checkpoint.pth` 已生成。

### 进行中
- 训练在进入 epoch 7 后因 CUDA OOM 退出，当前训练进程未运行。

### 待完成
- 从 `checkpoint.pth` 或 `checkpoint_0006.pth` Resume 继续训练。
- 若再次 OOM，先检查/清理其他显存占用；必要时再评估 AMP 或 batch size 调整。
## 2026-05-19 训练加速与控制台参数更新

### 已完成
- `tools/train.py` 已支持 dataloader 参数覆盖：`--train-batch-size`、`--val-batch-size`、`--train-num-workers`、`--val-num-workers`。
- `tools/training_dashboard.py` 已支持在网页里调整 Seed、AMP、训练/验证 batch size、训练/验证 num workers、checkpoint 保存间隔和命名风格。
- 控制台已新增“复原参数”按钮，一键恢复官方 R18 严格复现默认值：seed `42`、AMP 关闭、train batch size `4`、val batch size `8`、train/val num workers `4`、checkpoint step `1`、命名风格 `underscore`。
- 7861 控制台已重启，新控制台服务 PID 为 `59496`。
- 已用 in-app browser 验证新版 UI：参数输入框存在，点击“复原参数”后命令预览同步恢复默认参数。
- 已通过验证：`python -m unittest tests.test_checkpointing tests.test_training_dashboard tests.test_transforms_compat -v`。

### 进行中
- 当前训练未运行，准备从 epoch 6 的 `checkpoint.pth` 或 `checkpoint_0006.pth` resume。
- 因 epoch 7 曾出现 CUDA OOM，下一次 resume 前优先考虑是否开启 AMP，并记录它与严格官方默认参数的差异。

### 待完成
- 用户确认下一次 resume 使用默认严格复现参数，还是启用 AMP 做加速/降显存尝试。
- 继续监控 epoch 7/8 是否再次 OOM。
- 若修改 AMP、batch size 或 num workers，需要在 `复现指南.md` 继续记录参数与结果，避免和官方参考曲线混淆。
## 2026-05-19 控制台启动参数标签换行修正

### 已完成
- `tools/training_dashboard.py` 已修正启动参数区排版：`AMP` checkbox 独立成行，“训练 batch size”不再和 AMP 粘在同一行。
- `tests/test_training_dashboard.py` 已新增 `test_amp_control_is_on_its_own_row`，覆盖该布局约束。
- 已通过验证：`python -m unittest tests.test_training_dashboard -v`。
- 7861 控制台已重启，新控制台服务 PID 为 `59404`。
- 已用 in-app browser 验证页面上 `AMP` 行与“训练 batch size”标签分离。

### 进行中
- 控制台继续保持可用，用于下一次 resume 前确认训练参数和命令预览。

### 待完成
- 下一次启动训练前，继续确认参数预览没有和严格复现实验目标冲突。
## 2026-05-19 控制台命令预览标题换行修正

### 已完成
- `tools/training_dashboard.py` 已将“复原参数”按钮放入独立 `action-row`。
- “当前启动训练指令”和“当前 Resume 指令”已使用 `command-label`，确保在按钮下方换行显示。
- `tests/test_training_dashboard.py` 已新增 `test_command_preview_label_starts_after_action_row`。
- 已通过验证：`python -m unittest tests.test_training_dashboard -v`。
- 7861 控制台已重启，新控制台服务 PID 为 `59808`。
- 已用 in-app browser 验证按钮与“当前启动训练指令”已上下分离。

### 进行中
- 控制台继续用于下一次 resume 前确认训练参数和命令预览。

### 待完成
- 下一次启动训练前，继续确认是否启用 AMP，以及命令预览是否符合本次实验记录。
## 2026-05-19 控制台启动参数悬浮说明

### 已完成
- `tools/training_dashboard.py` 已给启动参数区添加 `help-label` / `data-help` tooltip。
- 配置文件、权重文件、输出目录、Seed、AMP、训练/验证 batch size、训练/验证 num workers、checkpoint 保存间隔、checkpoint 命名风格、当前启动训练指令、当前 Resume 指令均有悬浮解释。
- `tests/test_training_dashboard.py` 已新增 `test_training_parameter_labels_have_hover_help`。
- 已通过验证：`python -m unittest tests.test_training_dashboard -v`。
- 7861 控制台已重启，新控制台服务 PID 为 `57408`。
- 已用 in-app browser 验证页面存在 13 个带说明的标签，AMP 和训练 batch size 的说明文本已写入 DOM。

### 进行中
- 控制台继续用于下一次 resume 前确认参数含义、命令预览和复现实验设置。

### 待完成
- 若后续新增学习率、epoch 数等高风险训练参数，必须同步添加 hover 说明，并标注它们对严格复现可比性的影响。
## 2026-05-19 控制台布局与 UI 文案优化

### 已完成
- `tools/training_dashboard.py` 已将“启动参数”区域改为 `param-grid` 栅格布局，长路径参数独占一行，短数值参数两列排列。
- 命令预览已使用 `command-preview`，支持长命令自动换行，避免横向滚动。
- 主 dashboard 网格已补充 `min-width:0`，修复图表/日志区域可能撑出横向滚动条的问题。
- `tests/test_training_dashboard.py` 已新增 `test_startup_parameter_section_uses_optimized_layout`。
- 已通过验证：`python -m unittest tests.test_checkpointing tests.test_training_dashboard tests.test_transforms_compat -v`。
- 7861 控制台已重启，新控制台服务 PID 为 `56348`。
- 已用 in-app browser 验证无横向溢出，命令预览换行样式生效。

### 进行中
- 控制台继续用于下一次 resume 前检查参数、日志、checkpoint 和 GPU 状态。

### 待完成
- 下一次启动训练前，根据是否启用 AMP 更新实验记录，避免和严格默认参数曲线混淆。
