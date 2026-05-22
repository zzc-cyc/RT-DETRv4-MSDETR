# Git 工程管理

更新日期：2026-05-22

## 当前状态

### 已完成

* 已在 `D:\研究生\小论文` 初始化 Git 仓库。
* 已创建根目录 `.gitignore`，用于排除数据集、权重、预训练文件、checkpoint、评估缓存、压缩包、Python 缓存、本地浏览器/Agent 临时文件等。
* 已调整子项目 `.gitignore`：
  * `RT-DETR/RT-DETR-main/.gitignore` 不再统一忽略训练日志，并将 v1/v2 输出目录改为只忽略重型 `.pth`、`eval/`、`.pid` 等产物。
  * `RT-DETRv4/RT-DETRv4-main/.gitignore` 不再统一忽略训练日志。
* 已确认训练日志不被忽略，后续可以作为复现实验证据纳入 Git。
* 已确认权重、checkpoint、预训练模型、COCO 数据集和压缩包仍会被忽略。
* 已修正根 `.gitignore` 对 `dataset/datasets` 的误伤风险，保留 DINOv3、MrDETR、RT-DETRv4 中属于源码的 dataset 模块。
* 已排除 `RT-DETRv4/RT-DETRv4-main/dinov3/` junction，避免同一份 DINOv3 源码在 Git 中重复出现；真实源码只跟踪 `DINOv3/dinov3-main/`。
* 已新增 `.gitattributes`，固定常见源码/文档文本文件使用 LF 换行，减少 Windows/Linux 之间的换行噪音。

### 进行中

* 正在将初始化后的项目推送到 GitHub 仓库 `https://github.com/zzc-cyc/RT-DETRv4-MSDETR.git`。
* 本机未安装 GitHub CLI `gh`，本次发布使用原生 Git 命令完成。

### 待完成

* 推送完成后记录提交哈希和远程分支。
* 首次提交前再次运行暂存区大文件检查，确认没有误纳入权重或数据集。
* 后续每次实验结束后，将关键 `log.txt`、`console_train.log`、对比分析文档纳入版本管理；不要提交 `.pth`、COCO 图片/标注、预训练权重和压缩包。

## 跟踪策略

建议纳入 Git：

* 代码：`console/`、RT-DETR/RT-DETRv4/MrDETR/DINOv3 的源码改动。
* 配置：训练配置、数据集配置、控制台配置、实验开关。
* 文档：根目录 Markdown、论文复现分析、阶段报告、README。
* 日志：关键训练日志、官方参考日志、本地短训/对比日志。
* 小型论文和说明图片：用于阅读、复现报告和方法说明的 PDF/PNG/JPG。

不建议纳入 Git：

* COCO 数据集、下载数据集。
* `RT-DETRv4/RT-DETRv4-main/dinov3/` junction；对应真实源码在 `DINOv3/dinov3-main/`。
* `.pth`、`.pt`、`.pdparams`、`.onnx`、`.pkl`、`.npy`、`.npz` 等模型/缓存文件。
* `pretrain/`、`weights/`、`checkpoint/`、`checkpoints/` 下的权重。
* `eval/` 缓存、TensorBoard 事件、`wandb/`、临时测试输出。
* 项目压缩包、Python 缓存、浏览器和 Agent 本地状态文件。

## 常用检查命令

```powershell
git status --short --untracked-files=all
```

```powershell
git ls-files --others --exclude-standard
```

```powershell
git check-ignore -v "RT-DETRv4\RT-DETRv4-main\outputs\rtv4_hgnetv2_s_coco\last.pth"
```

```powershell
git check-ignore -v "RT-DETRv4\RT-DETRv4-main\outputs\rtv4_hgnetv2_s_coco\console_train.log"
```
