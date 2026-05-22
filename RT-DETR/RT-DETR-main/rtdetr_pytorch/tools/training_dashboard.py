import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
for path in (PROJECT_DIR, REPO_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

COCO_METRIC_NAMES = [
    "ap",
    "ap50",
    "ap75",
    "ap_small",
    "ap_medium",
    "ap_large",
    "ar1",
    "ar10",
    "ar100",
    "ar_small",
    "ar_medium",
    "ar_large",
]

DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output/rtdetr_r18vd_6x_coco"
DEFAULT_CONFIG = Path("configs/rtdetr/rtdetr_r18vd_6x_coco.yml")
DEFAULT_WEIGHT = Path("")
DEFAULT_PYTHON = Path(sys.executable)
PID_FILE = "dashboard_train.pid"
DEFAULT_TRAIN_BATCH_SIZE = 4
DEFAULT_VAL_BATCH_SIZE = 8
DEFAULT_TRAIN_NUM_WORKERS = 4
DEFAULT_VAL_NUM_WORKERS = 4


def default_training_params():
    return {
        "seed": 42,
        "amp": False,
        "train_batch_size": DEFAULT_TRAIN_BATCH_SIZE,
        "val_batch_size": DEFAULT_VAL_BATCH_SIZE,
        "train_num_workers": DEFAULT_TRAIN_NUM_WORKERS,
        "val_num_workers": DEFAULT_VAL_NUM_WORKERS,
        "checkpoint_step": 1,
        "checkpoint_name_style": "underscore",
    }


@dataclass
class DashboardConfig:
    project_dir: Path = PROJECT_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    python_exe: Path = DEFAULT_PYTHON
    config_path: Path = DEFAULT_CONFIG
    weight_path: Path = DEFAULT_WEIGHT
    seed: Optional[int] = 42
    amp: bool = False
    train_batch_size: Optional[int] = DEFAULT_TRAIN_BATCH_SIZE
    val_batch_size: Optional[int] = DEFAULT_VAL_BATCH_SIZE
    train_num_workers: Optional[int] = DEFAULT_TRAIN_NUM_WORKERS
    val_num_workers: Optional[int] = DEFAULT_VAL_NUM_WORKERS
    checkpoint_step: int = 1
    checkpoint_name_style: str = "underscore"
    pid: Optional[int] = None


def _resolve_project_path(config, path):
    path = Path(path)
    if str(path).strip() in ("", "."):
        return None
    if path.is_absolute():
        return path
    return config.project_dir / path


def _check_item(name, ok, detail):
    return {"name": name, "ok": bool(ok), "detail": str(detail)}


def _pid_is_running(pid):
    if not pid:
        return False
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0 and str(int(pid)) in result.stdout
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def _read_pid(output_dir):
    pid_path = Path(output_dir) / PID_FILE
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text(encoding="utf-8").strip().lstrip("\ufeff"))
    except ValueError:
        return None


def _write_pid(output_dir, pid):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / PID_FILE).write_text(str(pid), encoding="utf-8")


def _tail(path, limit=80):
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-limit:]


def _checkpoint_rows(output_dir):
    output_dir = Path(output_dir)
    checkpoint_paths = list(output_dir.glob("checkpoint*.pth"))
    numbered = [path for path in checkpoint_paths if path.name != "checkpoint.pth"]
    latest_pointer = [path for path in checkpoint_paths if path.name == "checkpoint.pth"]
    checkpoints = sorted(numbered, key=lambda p: p.stat().st_mtime, reverse=True) + latest_pointer
    return [
        {
            "name": path.name,
            "size_mb": round(path.stat().st_size / 1024 / 1024, 1),
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)),
        }
        for path in checkpoints
    ]


def _gpu_status():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    name, used, total, util = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
    return {
        "name": name,
        "memory_used_mb": int(used),
        "memory_total_mb": int(total),
        "utilization_gpu": int(util),
    }


def load_training_log(log_path):
    rows = []
    path = Path(log_path)
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        metrics = row.get("test_coco_eval_bbox") or []
        for name, value in zip(COCO_METRIC_NAMES, metrics):
            row[name] = value
        rows.append(row)
    return rows


def _rows_for_chart(rows):
    return [
        {
            "epoch": row.get("epoch"),
            "loss": row.get("train_loss"),
            "lr": row.get("train_lr"),
            "ap": row.get("ap"),
            "ap50": row.get("ap50"),
            "ap75": row.get("ap75"),
        }
        for row in rows
    ]


def build_train_command(config, resume=False):
    command = [
        str(config.python_exe),
        "tools/train.py",
        "-c",
        str(config.config_path).replace("\\", "/"),
    ]
    if resume:
        command.extend(["-r", str(Path(config.output_dir) / "checkpoint.pth")])
    elif config.weight_path and str(config.weight_path).strip() not in ("", "."):
        command.extend(["-t", str(config.weight_path).replace("\\", "/")])
    if config.seed is not None:
        command.extend(["--seed", str(config.seed)])
    if config.amp:
        command.append("--amp")
    if config.train_batch_size is not None:
        command.extend(["--train-batch-size", str(config.train_batch_size)])
    if config.val_batch_size is not None:
        command.extend(["--val-batch-size", str(config.val_batch_size)])
    if config.train_num_workers is not None:
        command.extend(["--train-num-workers", str(config.train_num_workers)])
    if config.val_num_workers is not None:
        command.extend(["--val-num-workers", str(config.val_num_workers)])
    command.extend(["--checkpoint-step", str(config.checkpoint_step)])
    command.extend(["--checkpoint-name-style", config.checkpoint_name_style])
    return command


def format_command(command):
    return " ".join(str(part) for part in command)


def log_open_mode(resume=False):
    return "a" if resume else "w"


def run_preflight_checks(config):
    output_dir = Path(config.output_dir)
    config_path = _resolve_project_path(config, config.config_path)
    python_exe = _resolve_project_path(config, config.python_exe)
    annotations_dir = Path(config.project_dir) / "dataset/coco/annotations"
    checks = [
        _check_item("Python environment", python_exe is None or python_exe.exists() or str(config.python_exe) == "python", config.python_exe),
        _check_item("config file", config_path is not None and config_path.exists(), config_path),
        _check_item("COCO train2017 image dir", (Path(config.project_dir) / "dataset/coco/train2017").exists(), Path(config.project_dir) / "dataset/coco/train2017"),
        _check_item("COCO val2017 image dir", (Path(config.project_dir) / "dataset/coco/val2017").exists(), Path(config.project_dir) / "dataset/coco/val2017"),
        _check_item("COCO train2017 annotation", (annotations_dir / "instances_train2017.json").exists(), annotations_dir / "instances_train2017.json"),
        _check_item("COCO val2017 annotation", (annotations_dir / "instances_val2017.json").exists(), annotations_dir / "instances_val2017.json"),
    ]

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        probe = output_dir / ".dashboard_write_check"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        checks.append(_check_item("output dir writable", True, output_dir))
    except OSError as exc:
        checks.append(_check_item("output dir writable", False, exc))

    current_pid = _read_pid(output_dir)
    checks.append(_check_item("no active dashboard training process", not _pid_is_running(current_pid), f"PID={current_pid}" if current_pid else "no training process detected"))
    return {"ok": all(item["ok"] for item in checks), "checks": checks}


def read_status(config):
    output_dir = Path(config.output_dir)
    rows = load_training_log(output_dir / "log.txt")
    latest = rows[-1] if rows else {}
    best = max(rows, key=lambda row: row.get("ap", -1), default={})
    pid = config.pid or _read_pid(output_dir)
    running = _pid_is_running(pid)
    return {
        "running": running,
        "pid": pid if running else None,
        "output_dir": str(output_dir),
        "config_path": str(config.config_path),
        "weight_path": str(config.weight_path),
        "python_exe": str(config.python_exe),
        "seed": config.seed,
        "amp": config.amp,
        "train_batch_size": config.train_batch_size,
        "val_batch_size": config.val_batch_size,
        "train_num_workers": config.train_num_workers,
        "val_num_workers": config.val_num_workers,
        "checkpoint_step": config.checkpoint_step,
        "checkpoint_name_style": config.checkpoint_name_style,
        "training_defaults": default_training_params(),
        "latest": latest,
        "best": best,
        "chart": _rows_for_chart(rows),
        "has_checkpoint": (output_dir / "checkpoint.pth").exists(),
        "checkpoints": _checkpoint_rows(output_dir),
        "console_tail": _tail(output_dir / "console_train.log"),
        "error_tail": _tail(output_dir / "console_train.err.log"),
        "gpu": _gpu_status(),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metric_names": COCO_METRIC_NAMES,
        "preflight": run_preflight_checks(config),
        "start_command": format_command(build_train_command(config, resume=False)),
        "resume_command": format_command(build_train_command(config, resume=True)),
    }


def start_training(config, resume=False):
    output_dir = Path(config.output_dir)
    current_pid = _read_pid(output_dir)
    if _pid_is_running(current_pid):
        return {"ok": False, "message": f"训练已在运行，PID={current_pid}"}
    if not resume:
        preflight = run_preflight_checks(config)
        if not preflight["ok"]:
            failed = [item["name"] for item in preflight["checks"] if not item["ok"]]
            return {"ok": False, "message": "正式训练前检查未通过：" + "、".join(failed), "preflight": preflight}

    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_train_command(config, resume=resume)
    mode = log_open_mode(resume=resume)
    stdout = (output_dir / "console_train.log").open(mode, encoding="utf-8")
    stderr = (output_dir / "console_train.err.log").open(mode, encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=config.project_dir,
        stdout=stdout,
        stderr=stderr,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    _write_pid(output_dir, process.pid)
    return {"ok": True, "message": f"训练已启动，PID={process.pid}", "pid": process.pid}


def stop_training(config):
    output_dir = Path(config.output_dir)
    pid = _read_pid(output_dir)
    if not _pid_is_running(pid):
        return {"ok": False, "message": "没有检测到由 dashboard 启动的运行中训练进程"}
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True, timeout=10)
        else:
            os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        return {"ok": False, "message": str(exc)}
    return {"ok": True, "message": f"已发送终止信号，PID={pid}"}


def _html():
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RT-DETR 训练控制台</title>
  <style>
    :root { --ink:#17202a; --muted:#64748b; --line:#d8dee9; --paper:#f6f8fb; --panel:#fff; --blue:#2563eb; --green:#059669; --orange:#ea580c; --red:#dc2626; }
    * { box-sizing: border-box; } body { margin:0; font-family: Arial, "Microsoft YaHei", sans-serif; color:var(--ink); background:var(--paper); }
    header { padding:20px 28px 14px; background:var(--panel); border-bottom:1px solid var(--line); display:flex; justify-content:space-between; gap:16px; align-items:flex-start; }
    h1 { margin:0 0 8px; font-size:26px; } .meta, label { color:var(--muted); font-size:13px; } label { display:block; }
    main { max-width:1440px; margin:0 auto; padding:20px 28px 36px; } .toolbar { display:flex; gap:10px; flex-wrap:wrap; }
    button { border:1px solid var(--line); background:var(--panel); color:var(--ink); border-radius:6px; padding:9px 12px; cursor:pointer; font-size:14px; }
    button.primary { background:var(--blue); color:white; border-color:var(--blue); } button.danger { background:var(--red); color:white; border-color:var(--red); } button:disabled { opacity:.45; cursor:not-allowed; }
    .stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin-bottom:16px; }
    .stat, section { background:var(--panel); border:1px solid var(--line); border-radius:8px; } .stat { padding:13px 14px; }
    .label { color:var(--muted); font-size:13px; margin-bottom:7px; } .value { font-size:22px; font-weight:700; word-break:break-word; }
    .grid { display:grid; grid-template-columns:minmax(0,1.25fr) minmax(360px,.75fr); gap:16px; } .grid > div { min-width:0; } section { min-width:0; overflow:hidden; padding:16px; margin-bottom:16px; } h2 { margin:0 0 12px; font-size:18px; }
    canvas { width:100%; height:280px; display:block; } pre { background:#0f172a; color:#dbeafe; padding:12px; border-radius:6px; overflow:auto; min-height:120px; max-height:280px; font-size:12px; line-height:1.5; }
    table { width:100%; border-collapse:collapse; font-size:13px; } th, td { border-bottom:1px solid var(--line); padding:8px; text-align:right; } th:first-child, td:first-child { text-align:left; }
    input, select { width:100%; border:1px solid var(--line); border-radius:6px; padding:8px; margin:4px 0 10px; background:white; }
    .param-grid { display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:10px 12px; }
    .field { min-width:0; }
    .field.full-span { grid-column:1 / -1; }
    .command-preview { white-space:pre-wrap; overflow-wrap:anywhere; min-height:88px; max-height:220px; }
    .checkbox-row { display:flex; align-items:center; gap:7px; margin:4px 0 12px; }
    .checkbox-row input { width:auto; margin:0; }
    .action-row { display:block; margin:14px 0 12px; }
    .command-label { display:block; margin-top:12px; clear:both; }
    .help-label { position:relative; cursor:help; text-decoration:underline dotted rgba(100,116,139,.65); text-underline-offset:3px; }
    .help-label:hover::after { content:attr(data-help); position:absolute; z-index:10; left:0; top:calc(100% + 6px); width:min(360px, calc(100vw - 48px)); padding:9px 10px; border:1px solid var(--line); border-radius:6px; background:#fff; color:var(--ink); box-shadow:0 8px 24px rgba(15,23,42,.16); line-height:1.5; font-size:12px; white-space:normal; }
    .status-dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; background:var(--red); } .status-dot.running { background:var(--green); }
    .check-list { display:grid; gap:8px; } .check-item { display:grid; grid-template-columns:24px minmax(120px,.75fr) minmax(0,1.25fr); gap:8px; align-items:start; padding:8px; border:1px solid var(--line); border-radius:6px; }
    .check-mark { width:18px; height:18px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center; color:white; font-size:12px; background:var(--red); } .check-mark.ok { background:var(--green); }
    .check-detail { color:var(--muted); word-break:break-all; font-size:12px; } @media (max-width:900px) { header { display:block; } .toolbar { margin-top:12px; } .grid { grid-template-columns:1fr; } .param-grid { grid-template-columns:1fr; } main, header { padding-left:16px; padding-right:16px; } }
  </style>
</head>
<body>
  <header><div><h1>RT-DETR 训练控制台</h1><div class="meta" id="subtitle">读取训练状态中...</div></div><div class="toolbar"><button class="primary" id="startBtn">启动训练</button><button id="resumeBtn">Resume</button><button class="danger" id="stopBtn">终止训练</button><button id="refreshBtn">刷新</button></div></header>
  <main><div class="stats" id="stats"></div><div class="grid"><div><section><h2>AP 曲线</h2><canvas id="apChart"></canvas></section><section><h2>Loss / LR 曲线</h2><canvas id="lossChart"></canvas></section><section><h2>Epoch 指标</h2><table id="metricsTable"></table></section></div><div><section><h2>正式训练前检查</h2><div class="meta" id="preflightSummary">等待检查...</div><div class="check-list" id="preflightChecks"></div></section><section><h2>启动参数</h2><div class="param-grid"><div class="field full-span"><label class="help-label" title="选择训练配置 YAML，决定模型结构、数据集、优化器和训练轮数。严格复现 R18 时保持 configs/rtdetr/rtdetr_r18vd_6x_coco.yml。" data-help="选择训练配置 YAML，决定模型结构、数据集、优化器和训练轮数。严格复现 R18 时保持 configs/rtdetr/rtdetr_r18vd_6x_coco.yml。">配置文件</label><input id="configPath"></div><div class="field"><label class="help-label" title="从头训练通常留空或为点号；使用 -t 会加载可匹配模型权重做 tuning。Resume 请使用下面的 Resume 按钮和 checkpoint.pth。" data-help="从头训练通常留空或为点号；使用 -t 会加载可匹配模型权重做 tuning。Resume 请使用下面的 Resume 按钮和 checkpoint.pth。">权重文件</label><input id="weightPath"></div><div class="field"><label class="help-label" title="随机种子，影响数据增强和初始化等随机过程。严格复现时固定为 42 并记录。" data-help="随机种子，影响数据增强和初始化等随机过程。严格复现时固定为 42 并记录。">Seed</label><input id="seed" type="number"></div><div class="field full-span"><label class="help-label" title="训练日志、checkpoint 和评估结果保存位置。当前 R18 复现实验输出到 output/rtdetr_r18vd_6x_coco。" data-help="训练日志、checkpoint 和评估结果保存位置。当前 R18 复现实验输出到 output/rtdetr_r18vd_6x_coco。">输出目录</label><input id="outputDir" disabled></div><div class="field"><label class="checkbox-row help-label" title="混合精度训练，可降低显存占用并可能加速；也可能让最终 AP 有轻微差异。开启后必须在实验记录中注明 AMP=True。" data-help="混合精度训练，可降低显存占用并可能加速；也可能让最终 AP 有轻微差异。开启后必须在实验记录中注明 AMP=True。"><input id="amp" type="checkbox">AMP</label></div><div class="field"><label class="help-label" title="每次训练迭代送入 GPU 的图片数。影响显存占用、速度和优化动态；官方 R18 单卡默认 4，严格复现建议保持 4。" data-help="每次训练迭代送入 GPU 的图片数。影响显存占用、速度和优化动态；官方 R18 单卡默认 4，严格复现建议保持 4。">训练 batch size</label><input id="trainBatchSize" type="number" min="1"></div><div class="field"><label class="help-label" title="验证阶段每个 batch 的图片数，只影响评估速度和显存占用，不直接改变训练结果。默认 8。" data-help="验证阶段每个 batch 的图片数，只影响评估速度和显存占用，不直接改变训练结果。默认 8。">验证 batch size</label><input id="valBatchSize" type="number" min="1"></div><div class="field"><label class="help-label" title="训练数据加载进程数。适当增大可能减少数据等待；Windows 上过高可能不稳定，建议 4、6、8 小步尝试。" data-help="训练数据加载进程数。适当增大可能减少数据等待；Windows 上过高可能不稳定，建议 4、6、8 小步尝试。">训练 num workers</label><input id="trainNumWorkers" type="number" min="0"></div><div class="field"><label class="help-label" title="验证数据加载进程数。影响验证吞吐和 CPU/内存占用，默认 4，过高可能增加系统负担。" data-help="验证数据加载进程数。影响验证吞吐和 CPU/内存占用，默认 4，过高可能增加系统负担。">验证 num workers</label><input id="valNumWorkers" type="number" min="0"></div><div class="field"><label class="help-label" title="每隔多少个 epoch 额外保留一个带轮次编号的 checkpoint。设为 1 表示每轮都保留，便于中断后从指定轮次恢复。" data-help="每隔多少个 epoch 额外保留一个带轮次编号的 checkpoint。设为 1 表示每轮都保留，便于中断后从指定轮次恢复。">Checkpoint 保存间隔 epoch</label><input id="checkpointStep" type="number" min="1"></div><div class="field"><label class="help-label" title="编号 checkpoint 的文件名格式。underscore 更直观，如 checkpoint_0006.pth；official 保持官方原始风格，如 checkpoint0006.pth。" data-help="编号 checkpoint 的文件名格式。underscore 更直观，如 checkpoint_0006.pth；official 保持官方原始风格，如 checkpoint0006.pth。">Checkpoint 命名风格</label><select id="checkpointNameStyle"><option value="underscore">checkpoint_0000.pth</option><option value="official">checkpoint0000.pth</option></select></div><div class="field full-span action-row"><button id="resetParamsBtn" type="button">复原参数</button></div><div class="field full-span"><label class="command-label help-label" title="根据当前表单参数生成的从头训练命令。点击启动训练前先核对这里，避免参数和实验目标不一致。" data-help="根据当前表单参数生成的从头训练命令。点击启动训练前先核对这里，避免参数和实验目标不一致。">当前启动训练指令</label><pre id="startCommandPreview" class="command-preview"></pre></div><div class="field full-span"><label class="command-label help-label" title="根据当前表单参数生成的断点续训命令，会从输出目录里的 checkpoint.pth 恢复。" data-help="根据当前表单参数生成的断点续训命令，会从输出目录里的 checkpoint.pth 恢复。">当前 Resume 指令</label><pre id="resumeCommandPreview" class="command-preview"></pre></div></div></section><section><h2>Checkpoint</h2><table id="checkpointTable"></table></section><section><h2>错误/警告日志</h2><pre id="errLog"></pre></section><section><h2>控制台日志</h2><pre id="consoleLog"></pre></section></div></div></main>
  <script>
    let state = null; const fmt = v => (v === undefined || v === null || v === '') ? '-' : (typeof v === 'number' ? v.toFixed(3) : v);
    async function api(path, body) { const res = await fetch(path, {method: body ? 'POST' : 'GET', headers:{'Content-Type':'application/json'}, body: body ? JSON.stringify(body) : undefined}); return await res.json(); }
    function numberOrNull(input) { return input.value === '' ? null : Number(input.value); }
    function params() { return {config_path:configPath.value, weight_path:weightPath.value, seed:numberOrNull(seed), amp:amp.checked, train_batch_size:numberOrNull(trainBatchSize), val_batch_size:numberOrNull(valBatchSize), train_num_workers:numberOrNull(trainNumWorkers), val_num_workers:numberOrNull(valNumWorkers), checkpoint_step:Number(checkpointStep.value || state.checkpoint_step || 1), checkpoint_name_style:checkpointNameStyle.value || state.checkpoint_name_style || 'underscore'}; }
    function commandPreview(resume=false) {
      const p = params();
      const parts = [state.python_exe, 'tools/train.py', '-c', p.config_path];
      if (resume) parts.push('-r', state.output_dir + '\\\\checkpoint.pth');
      else if (p.weight_path && p.weight_path.trim() && p.weight_path.trim() !== '.') parts.push('-t', p.weight_path);
      if (p.seed !== null) parts.push('--seed', String(p.seed));
      if (p.amp) parts.push('--amp');
      if (p.train_batch_size !== null) parts.push('--train-batch-size', String(p.train_batch_size));
      if (p.val_batch_size !== null) parts.push('--val-batch-size', String(p.val_batch_size));
      if (p.train_num_workers !== null) parts.push('--train-num-workers', String(p.train_num_workers));
      if (p.val_num_workers !== null) parts.push('--val-num-workers', String(p.val_num_workers));
      parts.push('--checkpoint-step', String(p.checkpoint_step));
      parts.push('--checkpoint-name-style', p.checkpoint_name_style);
      return parts.join(' ');
    }
    async function refresh() { state = await api('/api/status'); render(); }
    function render() {
      subtitle.innerHTML = `<span class="status-dot ${state.running ? 'running' : ''}"></span>${state.running ? '训练运行中' : '训练未运行'} · 更新 ${state.updated_at}`;
      configPath.value = state.config_path; weightPath.value = state.weight_path; outputDir.value = state.output_dir;
      if (!seed.value && state.seed !== null && state.seed !== undefined) seed.value = String(state.seed);
      if (!trainBatchSize.value && state.train_batch_size !== null && state.train_batch_size !== undefined) trainBatchSize.value = String(state.train_batch_size);
      if (!valBatchSize.value && state.val_batch_size !== null && state.val_batch_size !== undefined) valBatchSize.value = String(state.val_batch_size);
      if (!trainNumWorkers.value && state.train_num_workers !== null && state.train_num_workers !== undefined) trainNumWorkers.value = String(state.train_num_workers);
      if (!valNumWorkers.value && state.val_num_workers !== null && state.val_num_workers !== undefined) valNumWorkers.value = String(state.val_num_workers);
      amp.checked = Boolean(state.amp);
      if (!checkpointStep.value) checkpointStep.value = String(state.checkpoint_step || 1);
      if (!checkpointNameStyle.value) checkpointNameStyle.value = state.checkpoint_name_style || 'underscore';
      const preflightOk = state.preflight && state.preflight.ok; startBtn.disabled = state.running || !preflightOk; resumeBtn.disabled = state.running || !state.has_checkpoint; stopBtn.disabled = !state.running;
      renderPreflight(); const latest = state.latest || {}, best = state.best || {}, gpu = state.gpu || {};
      stats.innerHTML = [['PID',state.pid||'-'],['最新 Epoch',fmt(latest.epoch)],['最新 AP',fmt(latest.ap)],['最佳 AP',fmt(best.ap)],['最新 Loss',fmt(latest.train_loss)],['GPU',gpu.utilization_gpu!==undefined?gpu.utilization_gpu+'%':'-'],['显存',gpu.memory_used_mb?`${gpu.memory_used_mb}/${gpu.memory_total_mb} MB`:'-']].map(([k,v])=>`<div class="stat"><div class="label">${k}</div><div class="value">${v}</div></div>`).join('');
      metricsTable.innerHTML = '<thead><tr><th>Epoch</th><th>Loss</th><th>LR</th><th>AP</th><th>AP50</th><th>AP75</th></tr></thead><tbody>' + state.chart.slice(-12).reverse().map(r=>`<tr><td>${fmt(r.epoch)}</td><td>${fmt(r.loss)}</td><td>${fmt(r.lr)}</td><td>${fmt(r.ap)}</td><td>${fmt(r.ap50)}</td><td>${fmt(r.ap75)}</td></tr>`).join('') + '</tbody>';
      checkpointTable.innerHTML = '<thead><tr><th>文件</th><th>MB</th><th>时间</th></tr></thead><tbody>' + state.checkpoints.map(c=>`<tr><td>${c.name}</td><td>${c.size_mb}</td><td>${c.mtime}</td></tr>`).join('') + '</tbody>';
      errLog.textContent = state.error_tail.join('\\n') || '暂无错误日志'; consoleLog.textContent = state.console_tail.join('\\n') || '暂无控制台日志';
      updateCommandPreview();
      drawChart('apChart', [{name:'AP',color:'#2563eb',data:state.chart.map(r=>({x:r.epoch,y:r.ap})).filter(p=>p.y!==null&&p.y!==undefined)},{name:'AP50',color:'#059669',data:state.chart.map(r=>({x:r.epoch,y:r.ap50})).filter(p=>p.y!==null&&p.y!==undefined)},{name:'AP75',color:'#ea580c',data:state.chart.map(r=>({x:r.epoch,y:r.ap75})).filter(p=>p.y!==null&&p.y!==undefined)}], {minY:0,maxY:1});
      drawChart('lossChart', [{name:'loss',color:'#2563eb',data:state.chart.map(r=>({x:r.epoch,y:r.loss})).filter(p=>p.y!==null&&p.y!==undefined)},{name:'lr',color:'#ea580c',data:state.chart.map(r=>({x:r.epoch,y:r.lr})).filter(p=>p.y!==null&&p.y!==undefined)}]);
    }
    function updateCommandPreview(){ if(!state) return; startCommandPreview.textContent = commandPreview(false); resumeCommandPreview.textContent = commandPreview(true); }
    function restoreDefaultParams(){
      const d = state.training_defaults || {};
      seed.value = d.seed ?? 42;
      amp.checked = Boolean(d.amp);
      trainBatchSize.value = d.train_batch_size ?? 4;
      valBatchSize.value = d.val_batch_size ?? 8;
      trainNumWorkers.value = d.train_num_workers ?? 4;
      valNumWorkers.value = d.val_num_workers ?? 4;
      checkpointStep.value = d.checkpoint_step ?? 1;
      checkpointNameStyle.value = d.checkpoint_name_style || 'underscore';
      updateCommandPreview();
    }
    function renderPreflight(){ const p=state.preflight||{ok:false,checks:[]}; preflightSummary.textContent=p.ok?'全部检查通过，可以一键启动正式训练。':'检查未全部通过，启动按钮已锁定。'; preflightChecks.innerHTML=p.checks.map(i=>`<div class="check-item"><span class="check-mark ${i.ok?'ok':''}">${i.ok?'✓':'!'}</span><strong>${i.name}</strong><span class="check-detail">${i.detail}</span></div>`).join(''); }
    function drawXAxis(ctx, pad, plotW, plotH, minX, maxX, xScale, h) {
      const bottom = pad.top + plotH;
      const start = Math.ceil(minX);
      const end = Math.floor(maxX);
      const span = Math.max(1, end - start);
      const step = Math.max(1, Math.ceil(span / 6));
      ctx.fillStyle = '#64748b';
      ctx.strokeStyle = '#d8dee9';
      ctx.font = '12px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      for (let epoch = start; epoch <= end; epoch += step) {
        const x = xScale(epoch);
        ctx.beginPath();
        ctx.moveTo(x, bottom);
        ctx.lineTo(x, bottom + 5);
        ctx.stroke();
        ctx.fillText(String(epoch), x, bottom + 8);
      }
      if (end !== start && (end - start) % step !== 0) {
        const x = xScale(end);
        ctx.beginPath();
        ctx.moveTo(x, bottom);
        ctx.lineTo(x, bottom + 5);
        ctx.stroke();
        ctx.fillText(String(end), x, bottom + 8);
      }
      ctx.font = '12px Arial';
      ctx.fillText('Epoch', pad.left + plotW / 2, h - 35);
    }
    function drawChart(id, series, opts={}){ const canvas=document.getElementById(id),ctx=canvas.getContext('2d'),ratio=devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight; canvas.width=w*ratio; canvas.height=h*ratio; ctx.scale(ratio,ratio); ctx.clearRect(0,0,w,h); const points=series.flatMap(s=>s.data); if(!points.length){ctx.fillStyle='#64748b';ctx.fillText('暂无数据',40,32);return;} const pad={left:48,right:18,top:18,bottom:62},plotW=w-pad.left-pad.right,plotH=h-pad.top-pad.bottom,xs=points.map(p=>Number(p.x)),ys=points.map(p=>Number(p.y)),minX=Math.min(...xs),maxX=Math.max(...xs),rawMinY=Math.min(...ys),rawMaxY=Math.max(...ys),spanY=rawMaxY-rawMinY||1,minY=opts.minY??rawMinY-spanY*.08,maxY=opts.maxY??rawMaxY+spanY*.08,xScale=x=>pad.left+((x-minX)/((maxX-minX)||1))*plotW,yScale=y=>pad.top+(1-((y-minY)/((maxY-minY)||1)))*plotH; ctx.strokeStyle='#d8dee9'; ctx.beginPath(); ctx.moveTo(pad.left,pad.top); ctx.lineTo(pad.left,pad.top+plotH); ctx.lineTo(pad.left+plotW,pad.top+plotH); ctx.stroke(); ctx.fillStyle='#64748b'; ctx.font='12px Arial'; ctx.textAlign='right'; ctx.textBaseline='alphabetic'; for(let i=0;i<=4;i++){const y=minY+(maxY-minY)*i/4,py=yScale(y);ctx.fillText(y.toFixed(3),pad.left-8,py+4);ctx.strokeStyle='#edf1f5';ctx.beginPath();ctx.moveTo(pad.left,py);ctx.lineTo(pad.left+plotW,py);ctx.stroke();} drawXAxis(ctx,pad,plotW,plotH,minX,maxX,xScale,h); series.forEach(s=>{ctx.strokeStyle=s.color;ctx.lineWidth=2;ctx.beginPath();s.data.forEach((p,i)=>{const x=xScale(Number(p.x)),y=yScale(Number(p.y));if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});ctx.stroke();}); let lx=pad.left; series.forEach(s=>{ctx.fillStyle=s.color;ctx.fillRect(lx,h-18,10,10);ctx.fillStyle='#17202a';ctx.textAlign='left';ctx.textBaseline='alphabetic';ctx.fillText(s.name,lx+15,h-9);lx+=72;}); }
    startBtn.onclick=async()=>{alert((await api('/api/start',params())).message); refresh();}; resumeBtn.onclick=async()=>{alert((await api('/api/resume',params())).message); refresh();}; stopBtn.onclick=async()=>{if(confirm('确定终止当前训练进程？之后可用 checkpoint.pth resume。')){alert((await api('/api/stop',{})).message); refresh();}}; refreshBtn.onclick=refresh; resetParamsBtn.onclick=restoreDefaultParams; ['input','change'].forEach(evt=>{configPath.addEventListener(evt,updateCommandPreview);weightPath.addEventListener(evt,updateCommandPreview);seed.addEventListener(evt,updateCommandPreview);amp.addEventListener(evt,updateCommandPreview);trainBatchSize.addEventListener(evt,updateCommandPreview);valBatchSize.addEventListener(evt,updateCommandPreview);trainNumWorkers.addEventListener(evt,updateCommandPreview);valNumWorkers.addEventListener(evt,updateCommandPreview);checkpointStep.addEventListener(evt,updateCommandPreview);checkpointNameStyle.addEventListener(evt,updateCommandPreview);}); refresh(); setInterval(refresh,5000); addEventListener('resize',()=>state&&render());
  </script>
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    config: DashboardConfig = DashboardConfig()

    def _send_json(self, data, status=200):
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", "0"))
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            payload = _html().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        elif parsed.path == "/api/status":
            self._send_json(read_status(self.config))
        else:
            self._send_json({"ok": False, "message": "Not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        body = self._read_body()
        config = DashboardConfig(
            project_dir=self.config.project_dir,
            output_dir=self.config.output_dir,
            python_exe=self.config.python_exe,
            config_path=Path(body.get("config_path") or self.config.config_path),
            weight_path=Path(body.get("weight_path") or self.config.weight_path),
            seed=body.get("seed", self.config.seed),
            amp=bool(body.get("amp", self.config.amp)),
            train_batch_size=body.get("train_batch_size", self.config.train_batch_size),
            val_batch_size=body.get("val_batch_size", self.config.val_batch_size),
            train_num_workers=body.get("train_num_workers", self.config.train_num_workers),
            val_num_workers=body.get("val_num_workers", self.config.val_num_workers),
            checkpoint_step=int(body.get("checkpoint_step", self.config.checkpoint_step)),
            checkpoint_name_style=body.get("checkpoint_name_style", self.config.checkpoint_name_style),
        )
        if parsed.path == "/api/start":
            self._send_json(start_training(config, resume=False))
        elif parsed.path == "/api/resume":
            self._send_json(start_training(config, resume=True))
        elif parsed.path == "/api/stop":
            self._send_json(stop_training(config))
        else:
            self._send_json({"ok": False, "message": "Not found"}, 404)

    def log_message(self, format, *args):
        return


def run_server(config, host="127.0.0.1", port=7860):
    DashboardHandler.config = config
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"Dashboard: http://{host}:{port}")
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="Run a local RT-DETR training dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--weight", default=str(DEFAULT_WEIGHT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--train-batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--val-batch-size", type=int, default=DEFAULT_VAL_BATCH_SIZE)
    parser.add_argument("--train-num-workers", type=int, default=DEFAULT_TRAIN_NUM_WORKERS)
    parser.add_argument("--val-num-workers", type=int, default=DEFAULT_VAL_NUM_WORKERS)
    parser.add_argument("--checkpoint-step", type=int, default=1)
    parser.add_argument("--checkpoint-name-style", choices=["official", "underscore"], default="underscore")
    args = parser.parse_args()
    config = DashboardConfig(
        output_dir=Path(args.output_dir).resolve(),
        python_exe=Path(args.python),
        config_path=Path(args.config),
        weight_path=Path(args.weight),
        seed=args.seed,
        amp=args.amp,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        train_num_workers=args.train_num_workers,
        val_num_workers=args.val_num_workers,
        checkpoint_step=args.checkpoint_step,
        checkpoint_name_style=args.checkpoint_name_style,
    )
    run_server(config, args.host, args.port)


if __name__ == "__main__":
    main()
