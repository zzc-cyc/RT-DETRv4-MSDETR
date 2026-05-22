import tempfile
import unittest
from pathlib import Path

from src.solver.checkpointing import checkpoint_paths, epoch_checkpoint_name


class CheckpointingTests(unittest.TestCase):
    def test_underscore_epoch_checkpoint_name_is_explicit_resume_friendly(self):
        self.assertEqual(epoch_checkpoint_name(7, "underscore"), "checkpoint_0007.pth")

    def test_checkpoint_paths_keep_latest_pointer_and_epoch_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = checkpoint_paths(Path(tmp), epoch=1, checkpoint_step=1, name_style="underscore")

        self.assertEqual([path.name for path in paths], ["checkpoint.pth", "checkpoint_0001.pth"])

    def test_checkpoint_step_can_skip_numbered_epoch_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = checkpoint_paths(Path(tmp), epoch=1, checkpoint_step=5, name_style="underscore")

        self.assertEqual([path.name for path in paths], ["checkpoint.pth"])


if __name__ == "__main__":
    unittest.main()
