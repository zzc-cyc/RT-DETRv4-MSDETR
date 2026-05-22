"""Checkpoint naming helpers for training runs."""

from pathlib import Path


def epoch_checkpoint_name(epoch, style="official"):
    if style == "official":
        return f"checkpoint{epoch:04}.pth"
    if style == "underscore":
        return f"checkpoint_{epoch:04}.pth"
    raise ValueError(f"Unsupported checkpoint name style: {style}")


def checkpoint_paths(output_dir, epoch, checkpoint_step=1, name_style="official"):
    paths = [Path(output_dir) / "checkpoint.pth"]
    if checkpoint_step and (epoch + 1) % checkpoint_step == 0:
        paths.append(Path(output_dir) / epoch_checkpoint_name(epoch, name_style))
    return paths
