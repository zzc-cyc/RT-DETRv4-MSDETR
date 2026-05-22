import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXE = Path(sys.executable)
PID_TRAIN = "dashboard_train.pid"
PID_INFER = "dashboard_infer.pid"
COCO_ROOT = WORKSPACE_ROOT / "RT-DETR" / "RT-DETR-main" / "rtdetr_pytorch" / "dataset" / "coco"

# Cache official logs to avoid re-parsing multi-MB files on every 5s dashboard refresh.
# key: absolute path string -> (mtime, parsed_dict)
_OFFICIAL_V4_LOG_CACHE = {}


@dataclass(frozen=True)
class VersionProfile:
    key: str
    label: str
    repo_dir: Path
    train_script: Path
    infer_script: Path
    train_config: Path
    train_weight: Path
    output_dir: Path
    seed: int
    amp: bool
    train_batch_size: int
    val_batch_size: int
    train_num_workers: int
    val_num_workers: int
    checkpoint_step: Optional[int] = None
    checkpoint_name_style: Optional[str] = None
    use_total_batch_size: bool = False
    teacher_repo: Optional[Path] = None
    teacher_weight: Optional[Path] = None
    official_log: Optional[Path] = None


VERSION_PROFILES = {
    "v1": VersionProfile(
        key="v1",
        label="RT-DETR v1",
        repo_dir=WORKSPACE_ROOT / "RT-DETR" / "RT-DETR-main",
        train_script=Path("rtdetr_pytorch/tools/train.py"),
        infer_script=Path("rtdetr_pytorch/tools/infer.py"),
        train_config=Path("rtdetr_pytorch/configs/rtdetr/rtdetr_r18vd_6x_coco.yml"),
        train_weight=Path("weights/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth"),
        output_dir=WORKSPACE_ROOT / "RT-DETR" / "RT-DETR-main" / "rtdetr_pytorch" / "output" / "rtdetr_r18vd_6x_coco",
        seed=42,
        amp=False,
        train_batch_size=4,
        val_batch_size=8,
        train_num_workers=4,
        val_num_workers=4,
        checkpoint_step=1,
        checkpoint_name_style="underscore",
    ),
    "v4": VersionProfile(
        key="v4",
        label="RT-DETR v4",
        repo_dir=WORKSPACE_ROOT / "RT-DETRv4" / "RT-DETRv4-main",
        train_script=Path("train.py"),
        infer_script=Path("tools/inference/torch_inf.py"),
        train_config=Path("configs/rtv4/rtv4_hgnetv2_s_coco.yml"),
        train_weight=Path("checkpoint/RTv4-S-hgnet.pth"),
        output_dir=WORKSPACE_ROOT / "RT-DETRv4" / "RT-DETRv4-main" / "outputs" / "rtv4_hgnetv2_s_coco",
        seed=0,
        amp=True,
        train_batch_size=32,
        val_batch_size=64,
        train_num_workers=4,
        val_num_workers=4,
        use_total_batch_size=True,
        teacher_repo=Path("dinov3"),
        teacher_weight=Path("pretrain/dinov3_vitb16_pretrain_lvd1689m.pth"),
        official_log=Path("logs/RTv4-S-hgnet.log"),
    ),
}

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


def _profile(version: str) -> VersionProfile:
    return VERSION_PROFILES.get(version, VERSION_PROFILES["v4"])


def _sample_image() -> str:
    candidates = [
        WORKSPACE_ROOT / "RT-DETR" / "RT-DETR-main" / "rtdetr_pytorch" / "dataset" / "coco" / "val2017",
        WORKSPACE_ROOT / "RT-DETRv4" / "RT-DETRv4-main" / "dataset" / "coco" / "val2017",
    ]
    for folder in candidates:
        if folder.exists():
            for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                items = sorted(folder.glob(pattern))
                if items:
                    return str(items[0])
    return ""


def _resolve_path(base: Path, value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    raw = str(value).strip()
    if raw in ("", "."):
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = base / path
    return path


def _path_for_command(base: Path, path: Optional[Path]) -> str:
    if path is None:
        return ""
    if path.is_absolute():
        try:
            return str(path.relative_to(base)).replace("\\", "/")
        except ValueError:
            try:
                return os.path.relpath(path, base).replace("\\", "/")
            except ValueError:
                return str(path).replace("\\", "/")
    return str(path).replace("\\", "/")


def _json_bytes(data, status=200):
    return json.dumps(data, ensure_ascii=False).encode("utf-8"), status


def _check(name, ok, detail):
    return {"name": name, "ok": bool(ok), "detail": str(detail)}


def _pid_path(output_dir: Path, kind: str) -> Path:
    return output_dir / (PID_TRAIN if kind == "train" else PID_INFER)


def _is_running(pid: Optional[int]) -> bool:
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


def _read_pid(output_dir: Path, kind: str) -> Optional[int]:
    pid_file = _pid_path(output_dir, kind)
    if not pid_file.exists():
        return None
    try:
        return int(pid_file.read_text(encoding="utf-8").strip().lstrip("\ufeff"))
    except ValueError:
        return None


def _write_pid(output_dir: Path, kind: str, pid: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _pid_path(output_dir, kind).write_text(str(pid), encoding="utf-8")


def _tail(path: Path, limit=80):
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]


def load_epoch_times(console_log_path: Path):
    if not console_log_path.exists():
        return {}
    times = {}
    rx = re.compile(r"Epoch: \[(\d+)\] Total time: ([0-9:]+) \(")
    for line in console_log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = rx.search(line)
        if not m:
            continue
        times[int(m.group(1))] = m.group(2)
    return times


def _checkpoint_rows(output_dir: Path):
    if not output_dir.exists():
        return []
    checkpoints = [
        path for path in output_dir.glob("checkpoint*.pth")
        if path.name != "checkpoint.pth" or path.exists()
    ]
    for name in ("checkpoint.pth", "last.pth", "best_stg1.pth", "best_stg2.pth"):
        path = output_dir / name
        if path.exists() and path not in checkpoints:
            checkpoints.append(path)
    checkpoints = sorted(checkpoints, key=lambda p: p.stat().st_mtime, reverse=True)
    rows = []
    for path in checkpoints:
        try:
            rows.append({
                "name": path.name,
                "size_mb": round(path.stat().st_size / 1024 / 1024, 1),
                "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)),
            })
        except OSError:
            continue
    return rows


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


def load_training_log(log_path: Path):
    rows = []
    if not log_path.exists():
        return rows
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
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


def load_official_v4_log(log_path: Path):
    if not log_path.exists():
        return {"ok": False, "message": f"找不到官方日志文件: {log_path}", "epochs": {}}

    try:
        abs_key = str(log_path.resolve())
    except OSError:
        abs_key = str(log_path)
    try:
        mtime = log_path.stat().st_mtime
    except OSError:
        mtime = None

    cached = _OFFICIAL_V4_LOG_CACHE.get(abs_key)
    if cached and cached[0] == mtime:
        return cached[1]

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()

    rx_best = re.compile(r"best_stat: \{'epoch': (\d+), 'coco_eval_bbox': ([0-9.]+)\}")
    rx_last = re.compile(r"Epoch: \[(\d+)\]\s+\[\s*3695/3696\].*?loss: [0-9.]+ \(([0-9.]+)\)")
    rx_time = re.compile(r"Epoch: \[(\d+)\] Total time: ([0-9:]+) \(")

    # AP breakdown lines are printed after evaluation; we map them to the nearest preceding epoch total time.
    rx_epoch_total = re.compile(r"Epoch: \[(\d+)\] Total time:")
    rx_ap_all = re.compile(r"Average Precision\s+\(AP\) @\[ IoU=0\.50:0\.95 \| area=\s+all \| maxDets=100 \] = ([0-9.]+)")
    rx_ap50 = re.compile(r"Average Precision\s+\(AP\) @\[ IoU=0\.50\s+\| area=\s+all \| maxDets=100 \] = ([0-9.]+)")
    rx_ap75 = re.compile(r"Average Precision\s+\(AP\) @\[ IoU=0\.75\s+\| area=\s+all \| maxDets=100 \] = ([0-9.]+)")
    rx_aps = re.compile(r"Average Precision\s+\(AP\) @\[ IoU=0\.50:0\.95 \| area=\s+small \| maxDets=100 \] = ([0-9.]+)")
    rx_apm = re.compile(r"Average Precision\s+\(AP\) @\[ IoU=0\.50:0\.95 \| area=\s+medium \| maxDets=100 \] = ([0-9.]+)")
    rx_apl = re.compile(r"Average Precision\s+\(AP\) @\[ IoU=0\.50:0\.95 \| area=\s+large \| maxDets=100 \] = ([0-9.]+)")

    epochs = {}
    for line in lines:
        m = rx_best.search(line)
        if m:
            e = int(m.group(1))
            epochs.setdefault(e, {})["ap_best"] = float(m.group(2))
            continue
        m = rx_last.search(line)
        if m:
            e = int(m.group(1))
            epochs.setdefault(e, {})["avg_loss"] = float(m.group(2))
            continue
        m = rx_time.search(line)
        if m:
            e = int(m.group(1))
            epochs.setdefault(e, {})["time"] = m.group(2)
            continue

    cur_epoch = None
    cur = {}
    for line in lines:
        m = rx_epoch_total.search(line)
        if m:
            cur_epoch = int(m.group(1))
            cur = {}
            continue
        if cur_epoch is None:
            continue
        m = rx_ap_all.search(line)
        if m:
            cur["ap"] = float(m.group(1))
            continue
        m = rx_ap50.search(line)
        if m:
            cur["ap50"] = float(m.group(1))
            continue
        m = rx_ap75.search(line)
        if m:
            cur["ap75"] = float(m.group(1))
            continue
        m = rx_aps.search(line)
        if m:
            cur["aps"] = float(m.group(1))
            continue
        m = rx_apm.search(line)
        if m:
            cur["apm"] = float(m.group(1))
            continue
        m = rx_apl.search(line)
        if m:
            cur["apl"] = float(m.group(1))
            epochs.setdefault(cur_epoch, {}).update(cur)
            continue

    parsed = {"ok": True, "message": "ok", "epochs": epochs}
    _OFFICIAL_V4_LOG_CACHE[abs_key] = (mtime, parsed)
    return parsed


def build_official_compare(profile: VersionProfile, body: dict, rows: list, local_times: dict):
    if profile.key != "v4":
        return {"ok": False, "message": "当前仅支持 v4 官方日志对比", "path": "", "rows": []}

    repo_dir = profile.repo_dir
    requested = _resolve_path(repo_dir, body.get("official_log_path"))
    if requested is None and profile.official_log is not None:
        requested = repo_dir / profile.official_log
    if requested is None:
        return {"ok": False, "message": "未提供官方日志路径", "path": "", "rows": []}

    official = load_official_v4_log(requested)
    if not official["ok"]:
        return {"ok": False, "message": official.get("message", "读取官方日志失败"), "path": str(requested), "rows": []}

    out_rows = []
    epochs = official.get("epochs", {})
    for row in rows:
        e = row.get("epoch")
        if e is None:
            continue
        off = epochs.get(int(e), {})
        # Prefer official best_stat (ap_best). If missing for this epoch (common when best_stat didn't update),
        # fall back to the per-epoch evaluation AP parsed from the official log block.
        off_ap = off.get("ap_best")
        off_ap_source = "best_stat"
        if off_ap is None:
            off_ap = off.get("ap")
            off_ap_source = "epoch_ap" if off_ap is not None else "missing"
        local_ap = row.get("ap")
        delta = None
        if off_ap is not None and local_ap is not None:
            delta = float(local_ap) - float(off_ap)
        out_rows.append({
            "epoch": int(e),
            "local_ap": local_ap,
            # Backward-compat: this field used to mean "best_stat AP", but we now fall back to per-epoch AP
            # when best_stat did not update for this epoch.
            "official_ap_best": off_ap,
            # New canonical field name for the UI.
            "official_ap": off_ap,
            "official_ap_source": off_ap_source,
            "delta_ap": delta,
            "local_train_loss": row.get("train_loss"),
            "official_avg_loss": off.get("avg_loss"),
            "local_time": local_times.get(int(e)),
            "official_time": off.get("time"),
        })

    return {"ok": True, "message": "ok", "path": str(requested), "rows": out_rows}


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


def _train_defaults(profile: VersionProfile):
    if profile.use_total_batch_size:
        batch_label = "total_batch_size"
    else:
        batch_label = "batch_size"
    return {
        "seed": profile.seed,
        "amp": profile.amp,
        "train_batch_size": profile.train_batch_size,
        "val_batch_size": profile.val_batch_size,
        "train_num_workers": profile.train_num_workers,
        "val_num_workers": profile.val_num_workers,
        "checkpoint_step": profile.checkpoint_step,
        "checkpoint_name_style": profile.checkpoint_name_style,
        "batch_label": batch_label,
        "weight_path": "",
        "input_path": _sample_image(),
        "device": "cuda:0" if torch_available() else "cpu",
    }


def torch_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _infer_defaults():
    return {
        "input_path": _sample_image(),
        "device": "cuda:0" if torch_available() else "cpu",
    }


def _build_train_command(profile: VersionProfile, body: dict, action: str):
    repo_dir = profile.repo_dir
    train_entry = _path_for_command(repo_dir, repo_dir / profile.train_script)
    command = [str(PYTHON_EXE), train_entry]
    command.extend(["-c", _path_for_command(repo_dir, _resolve_path(repo_dir, body.get("config_path")) or (repo_dir / profile.train_config))])
    weight = _resolve_path(repo_dir, body.get("weight_path"))
    if action == "resume":
        output_dir = _resolve_path(repo_dir, body.get("output_dir")) or profile.output_dir
        resume_path = weight or (output_dir / ("last.pth" if profile.key == "v4" else "checkpoint.pth"))
        command.extend(["-r", _path_for_command(repo_dir, resume_path)])
    elif action == "eval":
        eval_weight = weight or _resolve_path(repo_dir, str(profile.train_weight))
        if eval_weight is not None:
            command.extend(["-r", _path_for_command(repo_dir, eval_weight)])
        else:
            command.extend(["-r", _path_for_command(repo_dir, (profile.output_dir / ("last.pth" if profile.key == "v4" else "checkpoint.pth")))])
        command.append("--test-only")
    elif weight is not None:
        command.extend(["-t", _path_for_command(repo_dir, weight)])

    seed = body.get("seed", profile.seed)
    if seed is not None and str(seed) != "":
        command.extend(["--seed", str(int(seed))])

    amp = bool(body.get("amp", profile.amp))
    if amp:
        command.append("--amp" if profile.key == "v1" else "--use-amp")

    output_dir = _resolve_path(repo_dir, body.get("output_dir")) or profile.output_dir
    v4_updates = []
    if profile.key == "v4":
        command.extend(["--output-dir", str(output_dir)])
        v4_updates.extend(_v4_dataset_overrides(repo_dir))
        # 对齐用户期望：每个 epoch 结束都保存一次 checkpoint000X.pth。
        # 注意：v4 的 last.pth 每轮都会更新；checkpoint_freq=1 会额外产生每轮一个大文件（约 170MB/轮）。
        v4_updates.append("checkpoint_freq=1")

    if profile.key == "v1":
        train_bs = body.get("train_batch_size", profile.train_batch_size)
        val_bs = body.get("val_batch_size", profile.val_batch_size)
        train_workers = body.get("train_num_workers", profile.train_num_workers)
        val_workers = body.get("val_num_workers", profile.val_num_workers)
        if train_bs is not None and str(train_bs) != "":
            command.extend(["--train-batch-size", str(int(train_bs))])
        if val_bs is not None and str(val_bs) != "":
            command.extend(["--val-batch-size", str(int(val_bs))])
        if train_workers is not None and str(train_workers) != "":
            command.extend(["--train-num-workers", str(int(train_workers))])
        if val_workers is not None and str(val_workers) != "":
            command.extend(["--val-num-workers", str(int(val_workers))])
        checkpoint_step = body.get("checkpoint_step")
        checkpoint_name_style = body.get("checkpoint_name_style")
        if checkpoint_step is not None and str(checkpoint_step) != "":
            command.extend(["--checkpoint-step", str(int(checkpoint_step))])
        if checkpoint_name_style:
            command.extend(["--checkpoint-name-style", str(checkpoint_name_style)])
    else:
        train_bs = body.get("train_batch_size", profile.train_batch_size)
        val_bs = body.get("val_batch_size", profile.val_batch_size)
        train_workers = body.get("train_num_workers", profile.train_num_workers)
        val_workers = body.get("val_num_workers", profile.val_num_workers)
        if train_bs is not None and str(train_bs) != "":
            v4_updates.append(f"train_dataloader.total_batch_size={int(train_bs)}")
        if val_bs is not None and str(val_bs) != "":
            v4_updates.append(f"val_dataloader.total_batch_size={int(val_bs)}")
        if train_workers is not None and str(train_workers) != "":
            v4_updates.append(f"train_dataloader.num_workers={int(train_workers)}")
        if val_workers is not None and str(val_workers) != "":
            v4_updates.append(f"val_dataloader.num_workers={int(val_workers)}")
        if v4_updates:
            command.extend(["-u", *v4_updates])

    return command, output_dir


def _build_infer_command(profile: VersionProfile, body: dict):
    repo_dir = profile.repo_dir
    command = [str(PYTHON_EXE), _path_for_command(repo_dir, repo_dir / profile.infer_script)]
    config = _resolve_path(repo_dir, body.get("config_path")) or (repo_dir / profile.train_config)
    command.extend(["-c", _path_for_command(repo_dir, config)])
    resume = _resolve_path(repo_dir, body.get("weight_path"))
    if resume is None:
        output_dir = _resolve_path(repo_dir, body.get("output_dir")) or profile.output_dir
        resume = _resolve_path(repo_dir, str(profile.train_weight)) or (output_dir / ("last.pth" if profile.key == "v4" else "checkpoint.pth"))
    command.extend(["-r", _path_for_command(repo_dir, resume)])
    input_path = _resolve_path(repo_dir, body.get("input_path"))
    if input_path is None:
        input_path = Path(_sample_image())
    flag = "-f" if profile.key == "v1" else "-i"
    command.extend([flag, _path_for_command(repo_dir, input_path)])
    device = str(body.get("device") or _infer_defaults()["device"])
    command.extend(["-d", device])
    return command


def _v4_dataset_overrides(repo_dir: Path):
    return [
        f"train_dataloader.dataset.img_folder={_path_for_command(repo_dir, COCO_ROOT / 'train2017')}",
        f"train_dataloader.dataset.ann_file={_path_for_command(repo_dir, COCO_ROOT / 'annotations' / 'instances_train2017.json')}",
        f"val_dataloader.dataset.img_folder={_path_for_command(repo_dir, COCO_ROOT / 'val2017')}",
        f"val_dataloader.dataset.ann_file={_path_for_command(repo_dir, COCO_ROOT / 'annotations' / 'instances_val2017.json')}",
    ]


def _run_preflight(profile: VersionProfile, body: dict, action: str, mode: str):
    repo_dir = profile.repo_dir
    config = _resolve_path(repo_dir, body.get("config_path")) or (repo_dir / profile.train_config)
    output_dir = _resolve_path(repo_dir, body.get("output_dir")) or profile.output_dir
    weight = _resolve_path(repo_dir, body.get("weight_path"))
    input_path = _resolve_path(repo_dir, body.get("input_path"))
    checks = [
        _check("repo dir", repo_dir.exists(), repo_dir),
        _check("train script", (repo_dir / profile.train_script).exists(), repo_dir / profile.train_script),
        _check("infer script", (repo_dir / profile.infer_script).exists(), repo_dir / profile.infer_script),
        _check("config file", config.exists(), config),
    ]

    checks.extend([
        _check("COCO train2017", (COCO_ROOT / "train2017").exists(), COCO_ROOT / "train2017"),
        _check("COCO val2017", (COCO_ROOT / "val2017").exists(), COCO_ROOT / "val2017"),
        _check("train annotations", (COCO_ROOT / "annotations" / "instances_train2017.json").exists(), COCO_ROOT / "annotations" / "instances_train2017.json"),
        _check("val annotations", (COCO_ROOT / "annotations" / "instances_val2017.json").exists(), COCO_ROOT / "annotations" / "instances_val2017.json"),
    ])

    if profile.key == "v4":
        teacher_repo = _resolve_path(repo_dir, str(profile.teacher_repo)) if profile.teacher_repo else None
        teacher_weight = _resolve_path(repo_dir, str(profile.teacher_weight)) if profile.teacher_weight else None
        hgnetv2_weight = repo_dir / "pretrain" / "hgnetv2" / "PPHGNetV2_B0_stage1.pth"
        checks.append(_check("DINOv3 repo", teacher_repo is not None and teacher_repo.exists(), teacher_repo or profile.teacher_repo))
        checks.append(_check("DINOv3 hubconf", teacher_repo is not None and (teacher_repo / "hubconf.py").exists(), (teacher_repo / "hubconf.py") if teacher_repo else profile.teacher_repo))
        checks.append(_check("DINOv3 weight", teacher_weight is not None and teacher_weight.exists(), teacher_weight or profile.teacher_weight))
        checks.append(_check("HGNetv2 B0 stage1 weight", hgnetv2_weight.exists(), hgnetv2_weight))
        teacher_text = config.read_text(encoding="utf-8", errors="replace") if config.exists() else ""
        checks.append(_check("teacher config section", "teacher_model" in teacher_text, config))

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        probe = output_dir / ".console_write_check"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        checks.append(_check("output dir writable", True, output_dir))
    except OSError as exc:
        checks.append(_check("output dir writable", False, exc))

    if mode == "infer":
        if weight is None:
            output_weight = _resolve_path(repo_dir, str(profile.train_weight)) or (output_dir / ("last.pth" if profile.key == "v4" else "checkpoint.pth"))
            checks.append(_check("inference checkpoint", output_weight.exists(), output_weight))
        else:
            checks.append(_check("inference checkpoint", weight.exists(), weight))
        checks.append(_check("input file", input_path is not None and input_path.exists(), input_path))
    elif action in {"resume", "eval"}:
        output_weight = weight or (_resolve_path(repo_dir, str(profile.train_weight)) if action == "eval" else None) or (output_dir / ("last.pth" if profile.key == "v4" else "checkpoint.pth"))
        if action == "resume":
            checks.append(_check("resume checkpoint", output_weight.exists(), output_weight))
        else:
            checks.append(_check("eval checkpoint", output_weight.exists(), output_weight))

    pid_file = _read_pid(output_dir, "train" if mode == "train" else "infer")
    checks.append(_check(f"{mode} process not already running", not _is_running(pid_file), f"PID={pid_file}" if pid_file else "no process"))
    return {"ok": all(item["ok"] for item in checks), "checks": checks}


def _status(profile: VersionProfile, body: dict, action: str, mode: str):
    repo_dir = profile.repo_dir
    output_dir = _resolve_path(repo_dir, body.get("output_dir")) or profile.output_dir
    config = _resolve_path(repo_dir, body.get("config_path")) or (repo_dir / profile.train_config)
    weight = _resolve_path(repo_dir, body.get("weight_path"))
    input_path = _resolve_path(repo_dir, body.get("input_path"))
    pid = _read_pid(output_dir, "train" if mode == "train" else "infer")
    running = _is_running(pid)
    rows = load_training_log(output_dir / "log.txt") if mode == "train" else []
    local_times = load_epoch_times(output_dir / "console_train.log") if mode == "train" else {}
    latest = rows[-1] if rows else {}
    best = max(rows, key=lambda row: row.get("ap", -1), default={}) if rows else {}
    train_command, _ = _build_train_command(profile, body, action if mode == "train" else "train")
    resume_command, _ = _build_train_command(profile, body, "resume")
    detect_command = _build_infer_command(profile, body)
    preflight = _run_preflight(profile, body, action, mode)
    has_checkpoint = any((output_dir / name).exists() for name in ("checkpoint.pth", "last.pth", "best_stg1.pth", "best_stg2.pth"))
    has_checkpoint = has_checkpoint or any(output_dir.glob("checkpoint*.pth"))
    return {
        "version": profile.key,
        "version_label": profile.label,
        "mode": mode,
        "action": action,
        "running": running,
        "pid": pid if running else None,
        "output_dir": str(output_dir),
        "config_path": str(config),
        "weight_path": str(weight) if weight else "",
        "input_path": str(input_path) if input_path else "",
        "seed": body.get("seed", profile.seed),
        "amp": bool(body.get("amp", profile.amp)),
        "train_batch_size": body.get("train_batch_size", profile.train_batch_size),
        "val_batch_size": body.get("val_batch_size", profile.val_batch_size),
        "train_num_workers": body.get("train_num_workers", profile.train_num_workers),
        "val_num_workers": body.get("val_num_workers", profile.val_num_workers),
        "checkpoint_step": body.get("checkpoint_step", profile.checkpoint_step),
        "checkpoint_name_style": body.get("checkpoint_name_style", profile.checkpoint_name_style),
        "training_defaults": _train_defaults(profile),
        "latest": latest,
        "best": best,
        "chart": _rows_for_chart(rows),
        "official_compare": build_official_compare(profile, body, rows, local_times) if mode == "train" else {"ok": False, "message": "仅训练模式可用", "path": "", "rows": []},
        "official_log_default": _path_for_command(repo_dir, (repo_dir / profile.official_log) if profile.official_log else None) if profile.key == "v4" else "",
        "has_checkpoint": has_checkpoint,
        "checkpoints": _checkpoint_rows(output_dir),
        "console_tail": _tail(output_dir / ("console_train.log" if mode == "train" else "console_infer.log")),
        "error_tail": _tail(output_dir / ("console_train.err.log" if mode == "train" else "console_infer.err.log")),
        "gpu": _gpu_status(),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metric_names": COCO_METRIC_NAMES,
        "preflight": preflight,
        "train_command": " ".join(train_command),
        "resume_command": " ".join(resume_command),
        "detect_command": " ".join(detect_command),
        "features": {
            "supports_checkpoint": profile.key == "v1",
            "supports_total_batch": profile.key == "v4",
            "supports_teacher_check": profile.key == "v4",
        },
    }


def _start_task(profile: VersionProfile, body: dict, action: str, mode: str):
    repo_dir = profile.repo_dir
    output_dir = _resolve_path(repo_dir, body.get("output_dir")) or profile.output_dir
    if mode == "train":
        preflight = _run_preflight(profile, body, action, mode)
        if not preflight["ok"]:
            failed = [item["name"] for item in preflight["checks"] if not item["ok"]]
            return {"ok": False, "message": "正式启动前检查未通过：" + "、".join(failed), "preflight": preflight}
        command, _ = _build_train_command(profile, body, action)
        cwd = repo_dir
        log_stdout = output_dir / "console_train.log"
        log_stderr = output_dir / "console_train.err.log"
        pid_kind = "train"
    else:
        preflight = _run_preflight(profile, body, action, mode)
        if not preflight["ok"]:
            failed = [item["name"] for item in preflight["checks"] if not item["ok"]]
            return {"ok": False, "message": "检测前检查未通过：" + "、".join(failed), "preflight": preflight}
        command = _build_infer_command(profile, body)
        cwd = repo_dir
        log_stdout = output_dir / "console_infer.log"
        log_stderr = output_dir / "console_infer.err.log"
        pid_kind = "infer"

    current_pid = _read_pid(output_dir, pid_kind)
    if _is_running(current_pid):
        return {"ok": False, "message": f"任务已在运行，PID={current_pid}"}

    output_dir.mkdir(parents=True, exist_ok=True)
    stdout = log_stdout.open("a" if action in {"resume"} else "w", encoding="utf-8")
    stderr = log_stderr.open("a" if action in {"resume"} else "w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=stdout,
        stderr=stderr,
        env=_subprocess_env(profile),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    _write_pid(output_dir, pid_kind, process.pid)
    label = "训练" if mode == "train" else "检测"
    return {"ok": True, "message": f"{label}已启动，PID={process.pid}", "pid": process.pid}


def _subprocess_env(profile: VersionProfile):
    env = os.environ.copy()
    if profile.key == "v4":
        env["USE_LIBUV"] = "0"
    return env


def _stop_task(profile: VersionProfile, body: dict, mode: str):
    repo_dir = profile.repo_dir
    output_dir = _resolve_path(repo_dir, body.get("output_dir")) or profile.output_dir
    pid_kind = "train" if mode == "train" else "infer"
    pid = _read_pid(output_dir, pid_kind)
    if not _is_running(pid):
        return {"ok": False, "message": "没有检测到运行中的任务"}
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True, timeout=10)
        else:
            os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        return {"ok": False, "message": str(exc)}
    return {"ok": True, "message": f"已发送终止信号，PID={pid}"}


def _read_body(handler):
    length = int(handler.headers.get("Content-Length", "0"))
    if not length:
        return {}
    return json.loads(handler.rfile.read(length).decode("utf-8"))


def _html():
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RT-DETR 统一控制台</title>
  <link rel="icon" href="data:,">
  <style>
    :root { --ink:#17202a; --muted:#64748b; --line:#d8dee9; --paper:#f6f8fb; --panel:#fff; --blue:#2563eb; --green:#059669; --orange:#ea580c; --red:#dc2626; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:Arial, "Microsoft YaHei", sans-serif; color:var(--ink); background:var(--paper); }
    header { padding:18px 24px 14px; background:var(--panel); border-bottom:1px solid var(--line); }
    .head-row { display:flex; align-items:flex-start; justify-content:space-between; gap:16px; flex-wrap:wrap; }
    h1 { margin:0 0 6px; font-size:24px; }
    .meta, label, .hint { color:var(--muted); font-size:13px; }
    main { max-width:1560px; margin:0 auto; padding:18px 24px 34px; }
    .toolbar { display:flex; gap:10px; flex-wrap:wrap; }
    button, select, input { border:1px solid var(--line); background:var(--panel); color:var(--ink); border-radius:6px; padding:9px 12px; font-size:14px; }
    button { cursor:pointer; }
    button.primary { background:var(--blue); color:#fff; border-color:var(--blue); }
    button.danger { background:var(--red); color:#fff; border-color:var(--red); }
    button.secondary { background:#eef2ff; border-color:#c7d2fe; }
    button:disabled { opacity:.45; cursor:not-allowed; }
    .nav { display:flex; gap:8px; align-items:center; margin-top:14px; flex-wrap:wrap; }
    .tab-btn { background:#eef2f7; border-color:#d5dde8; }
    .tab-btn.active { background:var(--blue); color:white; border-color:var(--blue); }
    .grid { display:grid; grid-template-columns:minmax(0,1.2fr) minmax(360px,.8fr); gap:16px; margin-top:16px; }
    .grid > div { min-width:0; }
    section, .card { background:var(--panel); border:1px solid var(--line); border-radius:8px; }
    section { min-width:0; overflow:hidden; padding:16px; margin-bottom:16px; }
    h2 { margin:0 0 12px; font-size:18px; }
    .stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin-bottom:16px; }
    .stat { padding:13px 14px; background:var(--panel); border:1px solid var(--line); border-radius:8px; }
    .label { color:var(--muted); font-size:13px; margin-bottom:6px; }
    .value { font-size:20px; font-weight:700; word-break:break-word; }
    .field-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px 12px; }
    .field { min-width:0; }
    .full { grid-column:1/-1; }
    .hidden { display:none !important; }
    .command-preview, pre { white-space:pre-wrap; overflow-wrap:anywhere; min-height:88px; max-height:260px; overflow:auto; background:#0f172a; color:#dbeafe; padding:12px; border-radius:6px; font-size:12px; line-height:1.5; }
    .check-list { display:grid; gap:8px; }
    .check-item { display:grid; grid-template-columns:22px minmax(120px,.7fr) minmax(0,1.3fr); gap:8px; align-items:start; padding:8px; border:1px solid var(--line); border-radius:6px; }
    .check-mark { width:18px; height:18px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center; color:white; font-size:12px; background:var(--red); }
    .check-mark.ok { background:var(--green); }
    .check-detail { color:var(--muted); word-break:break-all; font-size:12px; }
    .status-dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; background:var(--red); }
    .status-dot.running { background:var(--green); }
    .row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
    .subtle { color:var(--muted); font-size:12px; }
    .section-head { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:12px; }
    .section-head h2 { margin:0; }
    .mini-actions { display:flex; gap:8px; flex-wrap:wrap; }
    .mini-actions button { padding:6px 9px; font-size:12px; } 
    table { width:100%; border-collapse:collapse; font-size:13px; } 
    th, td { border-bottom:1px solid var(--line); padding:8px; text-align:right; } 
    th:first-child, td:first-child { text-align:left; } 
    .badge { display:inline-block; margin-left:6px; padding:2px 6px; border-radius:999px; background:#eef2f7; border:1px solid #d5dde8; color:#334155; font-size:11px; vertical-align:middle; } 
    .mode-panel { display:none; } 
    .mode-panel.active { display:block; } 
    @media (max-width:980px) {
      .grid { grid-template-columns:1fr; }
      .field-grid { grid-template-columns:1fr; }
      .check-item { grid-template-columns:22px 1fr; }
      .check-detail { grid-column:1/-1; padding-left:30px; }
      main, header { padding-left:16px; padding-right:16px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="head-row">
      <div>
        <h1>RT-DETR 统一控制台</h1>
        <div class="meta" id="subtitle">读取状态中...</div>
      </div>
      <div class="toolbar">
        <button class="secondary" id="refreshBtn">刷新</button>
      </div>
    </div>
    <div class="nav">
      <button class="tab-btn active" data-mode="train" id="trainTabBtn">训练</button>
      <button class="tab-btn" data-mode="infer" id="inferTabBtn">检测</button>
      <span class="hint">版本</span>
      <select id="versionSelect"></select>
    </div>
  </header>
  <main>
    <div class="stats" id="stats"></div>

    <section id="trainPanel" class="mode-panel active">
      <h2>训练 / 验证</h2>
      <div class="grid">
        <div>
          <section>
            <h2>运行状态</h2>
            <div class="subtle" id="trainCommandLabel">当前命令</div>
            <pre id="trainCommandPreview" class="command-preview"></pre>
            <div class="row" style="margin-top:12px;">
              <button class="primary" id="trainStartBtn">启动训练</button>
              <button id="trainResumeBtn">Resume</button>
              <button id="trainEvalBtn">仅验证</button>
              <button class="danger" id="trainStopBtn">终止训练</button>
            </div>
          </section>
          <section id="trainChartSection">
            <h2>AP 曲线</h2>
            <canvas id="apChart" style="width:100%;height:280px;display:block;"></canvas>
          </section>
          <section>
            <h2>Loss / LR 曲线</h2>
            <canvas id="lossChart" style="width:100%;height:280px;display:block;"></canvas>
          </section>
          <section>
            <h2>Epoch 指标</h2>
            <table id="metricsTable"></table>
          </section>
          <section class="v4-only" id="officialCompareSection">
            <div class="section-head">
              <h2>官方对比</h2>
              <div class="mini-actions">
                <button id="officialCompareRefreshBtn" type="button">刷新对比</button>
                <button id="officialCompareDefaultBtn" type="button">恢复默认</button>
              </div>
            </div>
            <div class="field-grid">
              <div class="field full">
                <label>官方训练日志文件</label>
                <input id="officialLogPath" placeholder="例如: logs/RTv4-S-hgnet.log">
                <div class="subtle" id="officialCompareHint"></div>
              </div>
            </div>
            <table id="officialCompareTable"></table>
          </section>
        </div>
        <div>
          <section>
            <h2>正式训练前检查</h2>
            <div class="meta" id="trainPreflightSummary">等待检查...</div>
            <div class="check-list" id="trainPreflightChecks"></div>
          </section>
          <section>
            <h2>启动参数</h2>
            <div class="field-grid">
              <div class="field full">
                <label id="configLabel">配置文件</label>
                <input id="configPath">
              </div>
              <div class="field">
                <label id="weightLabel">权重文件</label>
                <input id="weightPath">
              </div>
              <div class="field">
                <label>Seed</label>
                <input id="seed" type="number">
              </div>
              <div class="field full">
                <label>输出目录</label>
                <input id="outputDir">
              </div>
              <div class="field">
                <label id="ampLabel">AMP</label>
                <div class="row"><input id="amp" type="checkbox" style="width:auto;"> <span class="subtle">混合精度训练</span></div>
              </div>
              <div class="field">
                <label id="trainBatchLabel">训练 batch size</label>
                <input id="trainBatchSize" type="number" min="1">
              </div>
              <div class="field">
                <label id="valBatchLabel">验证 batch size</label>
                <input id="valBatchSize" type="number" min="1">
              </div>
              <div class="field">
                <label>训练 num workers</label>
                <input id="trainNumWorkers" type="number" min="0">
              </div>
              <div class="field">
                <label>验证 num workers</label>
                <input id="valNumWorkers" type="number" min="0">
              </div>
              <div class="field full v1-only">
                <label>Checkpoint 保存间隔 epoch</label>
                <input id="checkpointStep" type="number" min="1">
              </div>
              <div class="field full v1-only">
                <label>Checkpoint 命名风格</label>
                <select id="checkpointNameStyle">
                  <option value="underscore">checkpoint_0000.pth</option>
                  <option value="official">checkpoint0000.pth</option>
                </select>
              </div>
              <div class="field full">
                <label>当前训练指令</label>
                <pre id="trainCommandPreview2" class="command-preview"></pre>
              </div>
              <div class="field full">
                <label>当前 Resume 指令</label>
                <pre id="trainResumePreview" class="command-preview"></pre>
              </div>
            </div>
          </section>
          <section>
            <h2>Checkpoint</h2>
            <table id="checkpointTable"></table>
          </section>
          <section>
            <div class="section-head">
              <h2>错误 / 警告日志</h2>
              <div class="mini-actions">
                <button id="trainErrLogClearBtn" type="button">清空显示</button>
                <button id="trainErrLogRefreshBtn" type="button">刷新日志</button>
              </div>
            </div>
            <pre id="trainErrLog"></pre>
          </section>
          <section>
            <div class="section-head">
              <h2>控制台日志</h2>
              <div class="mini-actions">
                <button id="trainLogClearBtn" type="button">清空显示</button>
                <button id="trainLogRefreshBtn" type="button">刷新日志</button>
              </div>
            </div>
            <pre id="trainLog"></pre>
          </section>
        </div>
      </div>
    </section>

    <section id="inferPanel" class="mode-panel">
      <h2>检测 / 推理</h2>
      <div class="grid">
        <div>
          <section>
            <h2>运行状态</h2>
            <div class="subtle" id="inferCommandLabel">当前命令</div>
            <pre id="inferCommandPreview" class="command-preview"></pre>
            <div class="row" style="margin-top:12px;">
              <button class="primary" id="inferRunBtn">运行检测</button>
              <button class="danger" id="inferStopBtn">终止检测</button>
            </div>
          </section>
          <section>
            <h2>输入预览</h2>
            <div class="subtle">图像 / 视频输入由脚本自动判断，检测结果一般写入当前检测目录。</div>
            <div class="field-grid" style="margin-top:12px;">
              <div class="field full">
                <label>输入文件</label>
                <input id="inputPath">
              </div>
              <div class="field">
                <label>Device</label>
                <input id="device">
              </div>
            </div>
            <div class="field full" style="margin-top:12px;">
              <label>检测输出目录</label>
              <input id="inferOutputDir">
            </div>
            <div class="field full" style="margin-top:12px;">
              <label>当前检测指令</label>
              <pre id="inferCommandPreview2" class="command-preview"></pre>
            </div>
          </section>
          <section>
            <div class="section-head">
              <h2>错误 / 警告日志</h2>
              <div class="mini-actions">
                <button id="inferErrLogClearBtn" type="button">清空显示</button>
                <button id="inferErrLogRefreshBtn" type="button">刷新日志</button>
              </div>
            </div>
            <pre id="inferErrLog"></pre>
          </section>
          <section>
            <div class="section-head">
              <h2>控制台日志</h2>
              <div class="mini-actions">
                <button id="inferLogClearBtn" type="button">清空显示</button>
                <button id="inferLogRefreshBtn" type="button">刷新日志</button>
              </div>
            </div>
            <pre id="inferLog"></pre>
          </section>
        </div>
        <div>
          <section>
            <h2>检测前检查</h2>
            <div class="meta" id="inferPreflightSummary">等待检查...</div>
            <div class="check-list" id="inferPreflightChecks"></div>
          </section>
        </div>
      </div>
    </section>
  </main>
  <script>
    const VERSION_META = {
      v1: { label: 'RT-DETR v1', batchLabel: 'batch size', defaultMode: 'train' },
      v4: { label: 'RT-DETR v4', batchLabel: 'total batch size', defaultMode: 'train' },
    };
    let state = null;
    let mode = 'train';
    const logViewCleared = { trainLog:false, trainErrLog:false, inferLog:false, inferErrLog:false };

    const $ = id => document.getElementById(id);
    const fmt = v => (v === undefined || v === null || v === '') ? '-' : (typeof v === 'number' ? v.toFixed(3) : v);
    const q = (body) => fetch('/api/status', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body || {})}).then(r => r.json());
    const post = (path, body) => fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)}).then(r => r.json());
    const numberOrNull = el => el.value === '' ? null : Number(el.value);

    function currentForm() {
      return {
        version: $('versionSelect').value,
        mode,
        config_path: $('configPath').value,
        weight_path: $('weightPath').value,
        output_dir: $('outputDir').value || $('inferOutputDir').value,
        input_path: $('inputPath').value,
        device: $('device').value,
        seed: numberOrNull($('seed')),
        amp: $('amp').checked,
        train_batch_size: numberOrNull($('trainBatchSize')),
        val_batch_size: numberOrNull($('valBatchSize')),
        train_num_workers: numberOrNull($('trainNumWorkers')),
        val_num_workers: numberOrNull($('valNumWorkers')),
        checkpoint_step: numberOrNull($('checkpointStep')),
        checkpoint_name_style: $('checkpointNameStyle').value,
        official_log_path: $('officialLogPath') ? $('officialLogPath').value : '',
      };
    }

    function commandPreview(kind) {
      if (!state) return '';
      const key = kind === 'infer' ? 'detect_command' : 'train_command';
      return state[key] || '';
    }

    function renderTabs() {
      $('trainPanel').classList.toggle('active', mode === 'train');
      $('inferPanel').classList.toggle('active', mode === 'infer');
      $('trainTabBtn').classList.toggle('active', mode === 'train');
      $('inferTabBtn').classList.toggle('active', mode === 'infer');
    }

    function renderVersionSpecific() {
      const version = $('versionSelect').value;
      const meta = VERSION_META[version] || VERSION_META.v4;
      $('trainBatchLabel').textContent = version === 'v4' ? '训练 total batch size' : '训练 batch size';
      $('valBatchLabel').textContent = version === 'v4' ? '验证 total batch size' : '验证 batch size';
      document.querySelectorAll('.v1-only').forEach(el => el.classList.toggle('hidden', version !== 'v1'));
      document.querySelectorAll('.v4-only').forEach(el => el.classList.toggle('hidden', version !== 'v4'));
      $('trainCommandLabel').textContent = `${meta.label} 当前命令`;
      $('inferCommandLabel').textContent = `${meta.label} 当前命令`;
    }

    function renderPreflight(containerSummary, containerChecks, preflight) {
      const ok = preflight && preflight.ok;
      $(containerSummary).textContent = ok ? '全部检查通过，可以启动。' : '检查未全部通过，启动按钮会被锁定。';
      $(containerChecks).innerHTML = (preflight?.checks || []).map(item =>
        `<div class="check-item"><span class="check-mark ${item.ok ? 'ok' : ''}">${item.ok ? '✓' : '!'}</span><strong>${item.name}</strong><span class="check-detail">${item.detail}</span></div>`
      ).join('');
    }

    function renderStats() {
      const latest = state.latest || {};
      const best = state.best || {};
      const gpu = state.gpu || {};
      const rows = [
        ['版本', state.version_label || '-'],
        ['模式', state.mode || '-'],
        ['PID', state.pid || '-'],
        ['最新 Epoch', fmt(latest.epoch)],
        ['最新 AP', fmt(latest.ap)],
        ['最佳 AP', fmt(best.ap)],
        ['最新 Loss', fmt(latest.train_loss)],
        ['GPU', gpu.utilization_gpu !== undefined ? gpu.utilization_gpu + '%' : '-'],
      ];
      $('stats').innerHTML = rows.map(([k, v]) => `<div class="stat"><div class="label">${k}</div><div class="value">${v}</div></div>`).join('');
    }

    function renderTables() {
      const latestRows = (state.chart || []).slice(-12).reverse();
      $('metricsTable').innerHTML = '<thead><tr><th>Epoch</th><th>Loss</th><th>LR</th><th>AP</th><th>AP50</th><th>AP75</th></tr></thead><tbody>' +
        latestRows.map(row => `<tr><td>${fmt(row.epoch)}</td><td>${fmt(row.loss)}</td><td>${fmt(row.lr)}</td><td>${fmt(row.ap)}</td><td>${fmt(row.ap50)}</td><td>${fmt(row.ap75)}</td></tr>`).join('') + '</tbody>';
      $('checkpointTable').innerHTML = '<thead><tr><th>文件</th><th>MB</th><th>时间</th></tr></thead><tbody>' +
        (state.checkpoints || []).map(c => `<tr><td>${c.name}</td><td>${c.size_mb}</td><td>${c.mtime}</td></tr>`).join('') + '</tbody>';
    }

    function renderOfficialCompare() {
      const section = $('officialCompareSection');
      if (!section) return;
      const version = $('versionSelect').value;
      if (version !== 'v4') {
        $('officialCompareTable').innerHTML = '';
        $('officialCompareHint').textContent = '';
        return;
      }

      const key = 'officialLogPath_' + version;
      const input = $('officialLogPath');
      if (input && !input.value) {
        const saved = localStorage.getItem(key);
        input.value = saved || state.official_log_default || '';
      }

      const compare = state.official_compare || {};
      const rows = compare.rows || [];
      const hint = $('officialCompareHint');

      if (!compare.ok) {
        hint.textContent = compare.message ? ('官方对比不可用: ' + compare.message) : '官方对比不可用';
        $('officialCompareTable').innerHTML = '<thead><tr><th>提示</th></tr></thead><tbody><tr><td style=\"text-align:left;\">请检查官方日志路径是否存在，或点击“恢复默认”。</td></tr></tbody>';
        return;
      }

      hint.textContent = compare.path ? ('官方日志: ' + compare.path) : '';

      const fmt4 = v => (v === undefined || v === null || v === '') ? '-' : (Number(v).toFixed(4));
      const fmt2 = v => (v === undefined || v === null || v === '') ? '-' : (Number(v).toFixed(2));
      const fmtDelta = v => (v === undefined || v === null || v === '') ? '-' : ((v >= 0 ? '+' : '') + Number(v).toFixed(4));
      const fmtTime = v => (v === undefined || v === null || v === '') ? '-' : String(v);
      const fmtSource = s => {
        if (!s) return '';
        if (s === 'best_stat') return 'best';
        if (s === 'epoch_ap') return 'epoch';
        return String(s);
      };

      $('officialCompareTable').innerHTML =
        '<thead><tr>' +
          '<th>epoch</th>' +
          '<th>本地 AP</th>' +
          '<th>官方 AP</th>' +
          '<th>差值(本地-官方)</th>' +
          '<th>本地 train_loss</th>' +
          '<th>官方 avg_loss</th>' +
          '<th>本地耗时</th>' +
          '<th>官方耗时</th>' +
        '</tr></thead><tbody>' +
        rows.map(r => (
          `<tr>` +
            `<td>${fmt(r.epoch)}</td>` +
            `<td>${fmt4(r.local_ap)}</td>` +
            `<td>${fmt4(r.official_ap ?? r.official_ap_best)}${r.official_ap_source ? ' <span class=\"badge\">' + fmtSource(r.official_ap_source) + '</span>' : ''}</td>` +
            `<td>${fmtDelta(r.delta_ap)}</td>` +
            `<td>${fmt2(r.local_train_loss)}</td>` +
            `<td>${fmt2(r.official_avg_loss)}</td>` +
            `<td>${fmtTime(r.local_time)}</td>` +
            `<td>${fmtTime(r.official_time)}</td>` +
          `</tr>`
        )).join('') +
        '</tbody>';
    }

    function drawChart(canvasId, series, opts={}) {
      const canvas = document.getElementById(canvasId);
      const ctx = canvas.getContext('2d');
      const ratio = devicePixelRatio || 1;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      canvas.width = w * ratio;
      canvas.height = h * ratio;
      ctx.scale(ratio, ratio);
      ctx.clearRect(0, 0, w, h);
      const points = series.flatMap(s => s.data);
      if (!points.length) {
        ctx.fillStyle = '#64748b';
        ctx.fillText('暂无数据', 40, 32);
        return;
      }
      const pad = {left:48, right:18, top:18, bottom:62};
      const plotW = w - pad.left - pad.right;
      const plotH = h - pad.top - pad.bottom;
      const xs = points.map(p => Number(p.x));
      const ys = points.map(p => Number(p.y));
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const rawMinY = Math.min(...ys);
      const rawMaxY = Math.max(...ys);
      const spanY = rawMaxY - rawMinY || 1;
      const minY = opts.minY ?? rawMinY - spanY * .08;
      const maxY = opts.maxY ?? rawMaxY + spanY * .08;
      const xScale = x => pad.left + ((x - minX) / ((maxX - minX) || 1)) * plotW;
      const yScale = y => pad.top + (1 - ((y - minY) / ((maxY - minY) || 1))) * plotH;
      ctx.strokeStyle = '#d8dee9';
      ctx.beginPath();
      ctx.moveTo(pad.left, pad.top);
      ctx.lineTo(pad.left, pad.top + plotH);
      ctx.lineTo(pad.left + plotW, pad.top + plotH);
      ctx.stroke();
      ctx.fillStyle = '#64748b';
      ctx.font = '12px Arial';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'alphabetic';
      for (let i = 0; i <= 4; i++) {
        const y = minY + (maxY - minY) * i / 4;
        const py = yScale(y);
        ctx.fillText(y.toFixed(3), pad.left - 8, py + 4);
        ctx.strokeStyle = '#edf1f5';
        ctx.beginPath();
        ctx.moveTo(pad.left, py);
        ctx.lineTo(pad.left + plotW, py);
        ctx.stroke();
      }
      const start = Math.ceil(minX);
      const end = Math.floor(maxX);
      const span = Math.max(1, end - start);
      const step = Math.max(1, Math.ceil(span / 6));
      ctx.fillStyle = '#64748b';
      ctx.strokeStyle = '#d8dee9';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      for (let epoch = start; epoch <= end; epoch += step) {
        const x = xScale(epoch);
        ctx.beginPath();
        ctx.moveTo(x, pad.top + plotH);
        ctx.lineTo(x, pad.top + plotH + 5);
        ctx.stroke();
        ctx.fillText(String(epoch), x, h - 46);
      }
      ctx.fillText('Epoch', pad.left + plotW / 2, h - 28);
      series.forEach(s => {
        ctx.strokeStyle = s.color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        s.data.forEach((p, i) => {
          const x = xScale(Number(p.x));
          const y = yScale(Number(p.y));
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
      });
      let lx = pad.left;
      series.forEach(s => {
        ctx.fillStyle = s.color;
        ctx.fillRect(lx, h - 18, 10, 10);
        ctx.fillStyle = '#17202a';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'alphabetic';
        ctx.fillText(s.name, lx + 15, h - 8);
        lx += 72;
      });
    }

    function renderCharts() {
      drawChart('apChart', [
        {name:'AP', color:'#2563eb', data:(state.chart || []).map(r => ({x:r.epoch, y:r.ap})).filter(p => p.y !== null && p.y !== undefined)},
        {name:'AP50', color:'#059669', data:(state.chart || []).map(r => ({x:r.epoch, y:r.ap50})).filter(p => p.y !== null && p.y !== undefined)},
        {name:'AP75', color:'#ea580c', data:(state.chart || []).map(r => ({x:r.epoch, y:r.ap75})).filter(p => p.y !== null && p.y !== undefined)},
      ], {minY:0, maxY:1});
      drawChart('lossChart', [
        {name:'loss', color:'#2563eb', data:(state.chart || []).map(r => ({x:r.epoch, y:r.loss})).filter(p => p.y !== null && p.y !== undefined)},
        {name:'lr', color:'#ea580c', data:(state.chart || []).map(r => ({x:r.epoch, y:r.lr})).filter(p => p.y !== null && p.y !== undefined)},
      ]);
    }

    function renderLogs() {
      setLogText('trainLog', state.console_tail || [], '暂无控制台日志');
      setLogText('trainErrLog', state.error_tail || [], '暂无错误日志');
      setLogText('inferLog', state.console_tail || [], '暂无控制台日志');
      setLogText('inferErrLog', state.error_tail || [], '暂无错误日志');
    }

    function setLogText(logId, lines, emptyText) {
      if (logViewCleared[logId]) {
        $(logId).textContent = '已清空当前页面显示；点击“刷新日志”重新读取最新日志。';
        return;
      }
      $(logId).textContent = (lines || []).join('\\n') || emptyText;
    }

    function clearLogView(logId) {
      logViewCleared[logId] = true;
      $(logId).textContent = '已清空当前页面显示；点击“刷新日志”重新读取最新日志。';
    }

    function refreshLogView(logId) {
      logViewCleared[logId] = false;
      refresh();
    }

    function renderCommands() {
      if (!state) {
        $('trainCommandPreview').textContent = '';
        $('trainCommandPreview2').textContent = '';
        $('trainResumePreview').textContent = '';
        $('inferCommandPreview').textContent = '';
        $('inferCommandPreview2').textContent = '';
        return;
      }
      $('trainCommandPreview').textContent = state.train_command || '';
      $('trainCommandPreview2').textContent = state.train_command || '';
      $('trainResumePreview').textContent = state.resume_command || '';
      $('inferCommandPreview').textContent = state.detect_command || '';
      $('inferCommandPreview2').textContent = state.detect_command || '';
    }

    function render() {
      renderTabs();
      renderVersionSpecific();
      renderStats();
      renderTables();
      renderOfficialCompare();
      renderCommands();
      renderLogs();
      renderCharts();
      renderPreflight('trainPreflightSummary', 'trainPreflightChecks', state.preflight || {ok:false, checks:[]});
      renderPreflight('inferPreflightSummary', 'inferPreflightChecks', state.preflight || {ok:false, checks:[]});
      $('subtitle').innerHTML = `<span class="status-dot ${state.running ? 'running' : ''}"></span>${state.running ? '任务运行中' : '任务未运行'} · ${state.version_label || '-'} · 更新 ${state.updated_at || '-'}`;
      const features = state.features || {};
      $('checkpointStep').disabled = !features.supports_checkpoint;
      $('checkpointNameStyle').disabled = !features.supports_checkpoint;
      $('trainStartBtn').disabled = state.running || !state.preflight?.ok;
      $('trainResumeBtn').disabled = state.running || !state.has_checkpoint;
      $('trainEvalBtn').disabled = state.running || !state.has_checkpoint;
      $('trainStopBtn').disabled = !state.running;
      $('inferRunBtn').disabled = state.running || !state.preflight?.ok;
      $('inferStopBtn').disabled = !state.running;
      if (state.version === 'v4') {
        $('outputDir').disabled = false;
      } else {
        $('outputDir').disabled = true;
      }
    }

    async function refresh() {
      state = await q(currentForm());
      if (state.version) $('versionSelect').value = state.version;
      if (state.mode) mode = state.mode;
      const defaults = state.training_defaults || {};
      if (!$('seed').value) $('seed').value = defaults.seed ?? '';
      if (!$('amp').checked) $('amp').checked = Boolean(defaults.amp);
      if (!$('trainBatchSize').value) $('trainBatchSize').value = defaults.train_batch_size ?? '';
      if (!$('valBatchSize').value) $('valBatchSize').value = defaults.val_batch_size ?? '';
      if (!$('trainNumWorkers').value) $('trainNumWorkers').value = defaults.train_num_workers ?? '';
      if (!$('valNumWorkers').value) $('valNumWorkers').value = defaults.val_num_workers ?? '';
      if (!$('checkpointStep').value) $('checkpointStep').value = defaults.checkpoint_step ?? 1;
      if (!$('checkpointNameStyle').value) $('checkpointNameStyle').value = defaults.checkpoint_name_style || 'underscore';
      if (!$('inputPath').value) $('inputPath').value = defaults.input_path || '';
      if (!$('device').value) $('device').value = defaults.device || 'cpu';
      if (!$('configPath').value) $('configPath').value = state.config_path || '';
      if (!$('weightPath').value) $('weightPath').value = state.weight_path || '';
      if (!$('outputDir').value) $('outputDir').value = state.output_dir || '';
      if (!$('inferOutputDir').value) $('inferOutputDir').value = state.output_dir || '';
      if ($('officialLogPath') && !$('officialLogPath').value) {
        const v = state.version || $('versionSelect').value || 'v4';
        const saved = localStorage.getItem('officialLogPath_' + v);
        $('officialLogPath').value = saved || state.official_log_default || '';
      }
      render();
    }

    function restoreDefaults() {
      const version = $('versionSelect').value;
      const defaults = (state && state.training_defaults) ? state.training_defaults : {
        seed: version === 'v1' ? 42 : 0,
        amp: version === 'v4',
        train_batch_size: version === 'v4' ? 32 : 4,
        val_batch_size: version === 'v4' ? 64 : 8,
        train_num_workers: 4,
        val_num_workers: 4,
        checkpoint_step: 1,
        checkpoint_name_style: 'underscore',
        input_path: '',
        device: 'cpu',
      };
      $('seed').value = defaults.seed ?? '';
      $('amp').checked = Boolean(defaults.amp);
      $('trainBatchSize').value = defaults.train_batch_size ?? '';
      $('valBatchSize').value = defaults.val_batch_size ?? '';
      $('trainNumWorkers').value = defaults.train_num_workers ?? '';
      $('valNumWorkers').value = defaults.val_num_workers ?? '';
      $('checkpointStep').value = defaults.checkpoint_step ?? 1;
      $('checkpointNameStyle').value = defaults.checkpoint_name_style || 'underscore';
      $('inputPath').value = defaults.input_path || '';
      $('device').value = defaults.device || 'cpu';
      if ($('officialLogPath')) {
        const saved = localStorage.getItem('officialLogPath_' + version);
        const fallback = (state && state.official_log_default) ? state.official_log_default : '';
        $('officialLogPath').value = saved || fallback || '';
      }
      renderVersionSpecific();
      renderCommands();
    }

    function resetVersionFields() {
      ['configPath','weightPath','outputDir','inferOutputDir','inputPath','device','seed','trainBatchSize','valBatchSize','trainNumWorkers','valNumWorkers','checkpointStep','officialLogPath'].forEach(id => {
        $(id).value = '';
      });
      $('amp').checked = false;
      $('checkpointNameStyle').value = 'underscore';
    }

    function currentAction(btnId) {
      if (btnId === 'trainResumeBtn') return 'resume';
      if (btnId === 'trainEvalBtn') return 'eval';
      return 'train';
    }

    $('versionSelect').innerHTML = Object.entries(VERSION_META).map(([k, v]) => `<option value="${k}">${v.label}</option>`).join('');
    $('versionSelect').value = 'v4';

    $('trainTabBtn').onclick = () => { mode = 'train'; refresh(); };
    $('inferTabBtn').onclick = () => { mode = 'infer'; refresh(); };
    $('refreshBtn').onclick = refresh;
    if ($('officialCompareRefreshBtn')) $('officialCompareRefreshBtn').onclick = refresh;
    if ($('officialCompareDefaultBtn')) $('officialCompareDefaultBtn').onclick = () => {
      const version = $('versionSelect').value || 'v4';
      const v = (state && state.official_log_default) ? state.official_log_default : '';
      $('officialLogPath').value = v;
      localStorage.setItem('officialLogPath_' + version, v);
      refresh();
    };
    if ($('officialLogPath')) {
      $('officialLogPath').addEventListener('change', () => {
        const version = $('versionSelect').value || 'v4';
        localStorage.setItem('officialLogPath_' + version, $('officialLogPath').value || '');
        refresh();
      });
    }
    $('versionSelect').onchange = () => { resetVersionFields(); renderVersionSpecific(); refresh(); };

    async function startTraining(action) {
      const payload = currentForm();
      payload.action = action;
      const res = await post('/api/run', payload);
      alert(res.message || '完成');
      refresh();
    }

    $('trainStartBtn').onclick = () => startTraining('train');
    $('trainResumeBtn').onclick = () => startTraining('resume');
    $('trainEvalBtn').onclick = () => startTraining('eval');
    $('trainStopBtn').onclick = async () => { if (confirm('确定终止当前训练进程？')) { alert((await post('/api/stop', {...currentForm(), mode:'train'})).message); refresh(); } };
    $('inferRunBtn').onclick = async () => { const res = await post('/api/run', {...currentForm(), action:'infer', mode:'infer'}); alert(res.message || '完成'); refresh(); };
    $('inferStopBtn').onclick = async () => { if (confirm('确定终止当前检测进程？')) { alert((await post('/api/stop', {...currentForm(), mode:'infer'})).message); refresh(); } };
    $('trainLogClearBtn').onclick = () => clearLogView('trainLog');
    $('trainLogRefreshBtn').onclick = () => refreshLogView('trainLog');
    $('trainErrLogClearBtn').onclick = () => clearLogView('trainErrLog');
    $('trainErrLogRefreshBtn').onclick = () => refreshLogView('trainErrLog');
    $('inferLogClearBtn').onclick = () => clearLogView('inferLog');
    $('inferLogRefreshBtn').onclick = () => refreshLogView('inferLog');
    $('inferErrLogClearBtn').onclick = () => clearLogView('inferErrLog');
    $('inferErrLogRefreshBtn').onclick = () => refreshLogView('inferErrLog');

    ['input','change'].forEach(evt => {
      ['configPath','weightPath','outputDir','inputPath','device','seed','amp','trainBatchSize','valBatchSize','trainNumWorkers','valNumWorkers','checkpointStep','checkpointNameStyle'].forEach(id => {
        $(id).addEventListener(evt, () => { renderCommands(); });
      });
    });

    restoreDefaults();
    refresh();
    setInterval(refresh, 5000);
    addEventListener('resize', () => state && renderCharts());
  </script>
</body>
</html>"""


class ConsoleHandler(BaseHTTPRequestHandler):
    def _send_json(self, data, status=200):
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            payload = _html().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self._send_json({"ok": False, "message": "Not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        body = _read_body(self)
        profile = _profile(body.get("version", "v4"))
        action = body.get("action", "train")
        mode = body.get("mode", "train")
        if parsed.path == "/api/status":
            self._send_json(_status(profile, body, action, mode))
            return
        if parsed.path == "/api/run":
            self._send_json(_start_task(profile, body, action, mode))
            return
        if parsed.path == "/api/stop":
            self._send_json(_stop_task(profile, body, mode))
            return
        self._send_json({"ok": False, "message": "Not found"}, 404)

    def log_message(self, format, *args):
        return


def run_server(host="127.0.0.1", port=7860):
    server = ThreadingHTTPServer((host, port), ConsoleHandler)
    print(f"Dashboard: http://{host}:{port}")
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="Run the unified RT-DETR console.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
