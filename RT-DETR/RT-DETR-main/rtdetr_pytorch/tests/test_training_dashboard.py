import json
import tempfile
import unittest
from pathlib import Path

from tools import training_dashboard as dashboard


class TrainingDashboardTests(unittest.TestCase):
    def test_load_training_log_maps_coco_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "log.txt"
            row = {
                "epoch": 3,
                "train_loss": 10.5,
                "train_lr": 0.0001,
                "test_coco_eval_bbox": [0.46, 0.63, 0.50, 0.28, 0.49, 0.62, 0.36, 0.61, 0.68, 0.49, 0.73, 0.85],
            }
            log_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            rows = dashboard.load_training_log(log_path)

        self.assertEqual(rows[0]["epoch"], 3)
        self.assertEqual(rows[0]["train_loss"], 10.5)
        self.assertEqual(rows[0]["ap"], 0.46)
        self.assertEqual(rows[0]["ap50"], 0.63)
        self.assertEqual(rows[0]["ar_large"], 0.85)

    def test_build_train_command_uses_official_from_scratch_flow(self):
        cfg = dashboard.DashboardConfig(
            project_dir=Path("project"),
            output_dir=Path("project/output/rtdetr_r18vd_6x_coco"),
            python_exe=Path("python"),
            config_path=Path("configs/rtdetr/rtdetr_r18vd_6x_coco.yml"),
            weight_path=Path(""),
            seed=42,
            amp=True,
            train_batch_size=4,
            val_batch_size=8,
            train_num_workers=6,
            val_num_workers=4,
        )

        command = dashboard.build_train_command(cfg, resume=False)

        self.assertEqual(command[:4], ["python", "tools/train.py", "-c", "configs/rtdetr/rtdetr_r18vd_6x_coco.yml"])
        self.assertNotIn("-t", command)
        self.assertIn("--seed", command)
        self.assertIn("--amp", command)
        self.assertIn("--checkpoint-step", command)
        self.assertIn("--checkpoint-name-style", command)
        self.assertIn("underscore", command)
        self.assertIn("--train-batch-size", command)
        self.assertIn("4", command)
        self.assertIn("--val-batch-size", command)
        self.assertIn("8", command)
        self.assertIn("--train-num-workers", command)
        self.assertIn("6", command)
        self.assertIn("--val-num-workers", command)
        self.assertIn("4", command)

    def test_train_overrides_build_dataloader_kwargs(self):
        from tools import train

        args = type(
            "Args",
            (),
            {
                "train_batch_size": 3,
                "val_batch_size": 5,
                "train_num_workers": 6,
                "val_num_workers": 2,
            },
        )()

        overrides = train.build_dataloader_overrides(args)

        self.assertEqual(overrides["train_dataloader"]["batch_size"], 3)
        self.assertEqual(overrides["val_dataloader"]["batch_size"], 5)
        self.assertEqual(overrides["train_dataloader"]["num_workers"], 6)
        self.assertEqual(overrides["val_dataloader"]["num_workers"], 2)

    def test_default_training_params_match_official_r18_reproduction(self):
        defaults = dashboard.default_training_params()

        self.assertEqual(defaults["seed"], 42)
        self.assertFalse(defaults["amp"])
        self.assertEqual(defaults["train_batch_size"], 4)
        self.assertEqual(defaults["val_batch_size"], 8)
        self.assertEqual(defaults["train_num_workers"], 4)
        self.assertEqual(defaults["val_num_workers"], 4)
        self.assertEqual(defaults["checkpoint_step"], 1)
        self.assertEqual(defaults["checkpoint_name_style"], "underscore")

    def test_read_status_includes_command_previews(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "configs/rtdetr").mkdir(parents=True)
            (project / "configs/rtdetr/rtdetr_r18vd_6x_coco.yml").write_text("task: detection\n", encoding="utf-8")
            output = project / "output/rtdetr_r18vd_6x_coco"
            cfg = dashboard.DashboardConfig(
                project_dir=project,
                output_dir=output,
                python_exe=Path("python"),
                config_path=Path("configs/rtdetr/rtdetr_r18vd_6x_coco.yml"),
                seed=7,
            )

            status = dashboard.read_status(cfg)

        self.assertIn("start_command", status)
        self.assertIn("resume_command", status)
        self.assertIn("--seed 7", status["start_command"])
        self.assertIn("-r", status["resume_command"])
        self.assertIn("training_defaults", status)

    def test_preflight_requires_coco_layout_and_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "configs/rtdetr").mkdir(parents=True)
            (project / "configs/rtdetr/rtdetr_r18vd_6x_coco.yml").write_text("task: detection\n", encoding="utf-8")
            (project / "dataset/coco/train2017").mkdir(parents=True)
            (project / "dataset/coco/val2017").mkdir(parents=True)
            (project / "dataset/coco/annotations").mkdir(parents=True)
            (project / "dataset/coco/annotations/instances_train2017.json").write_text("{}", encoding="utf-8")
            (project / "dataset/coco/annotations/instances_val2017.json").write_text("{}", encoding="utf-8")
            output = project / "output/rtdetr_r18vd_6x_coco"
            cfg = dashboard.DashboardConfig(
                project_dir=project,
                output_dir=output,
                python_exe=Path("python"),
                config_path=Path("configs/rtdetr/rtdetr_r18vd_6x_coco.yml"),
            )

            result = dashboard.run_preflight_checks(cfg)

        self.assertTrue(result["ok"])
        self.assertGreaterEqual(len(result["checks"]), 6)

    def test_log_open_mode_overwrites_for_fresh_start_and_appends_for_resume(self):
        self.assertEqual(dashboard.log_open_mode(resume=False), "w")
        self.assertEqual(dashboard.log_open_mode(resume=True), "a")

    def test_chart_html_labels_x_axis_as_epoch(self):
        html = dashboard._html()

        self.assertIn("function drawXAxis", html)
        self.assertIn("Epoch", html)

    def test_html_includes_training_parameter_controls_and_reset_button(self):
        html = dashboard._html()

        self.assertIn("trainBatchSize", html)
        self.assertIn("valBatchSize", html)
        self.assertIn("trainNumWorkers", html)
        self.assertIn("valNumWorkers", html)
        self.assertIn("resetParamsBtn", html)

    def test_amp_control_is_on_its_own_row(self):
        html = dashboard._html()

        self.assertIn('class="checkbox-row help-label"', html)
        self.assertIn("label { display:block;", html)

    def test_command_preview_label_starts_after_action_row(self):
        html = dashboard._html()

        self.assertIn('class="field full-span action-row"', html)
        self.assertIn('class="command-label help-label"', html)
        self.assertIn(".command-label { display:block;", html)

    def test_training_parameter_labels_have_hover_help(self):
        html = dashboard._html()

        expected_help_targets = [
            "配置文件",
            "权重文件",
            "输出目录",
            "Seed",
            "AMP",
            "训练 batch size",
            "验证 batch size",
            "训练 num workers",
            "验证 num workers",
            "Checkpoint 保存间隔 epoch",
            "Checkpoint 命名风格",
        ]
        for target in expected_help_targets:
            self.assertIn(target, html)

        self.assertIn("help-label", html)
        self.assertIn("data-help", html)
        self.assertIn(".help-label:hover::after", html)
        self.assertIn("影响显存占用", html)
        self.assertIn("严格复现", html)

    def test_startup_parameter_section_uses_optimized_layout(self):
        html = dashboard._html()

        self.assertIn('class="param-grid"', html)
        self.assertIn('class="field"', html)
        self.assertIn('class="field full-span"', html)
        self.assertIn('class="command-preview"', html)
        self.assertIn("white-space:pre-wrap", html)
        self.assertIn("overflow-wrap:anywhere", html)
        self.assertIn(".grid > div { min-width:0;", html)
        self.assertIn("section { min-width:0;", html)


if __name__ == "__main__":
    unittest.main()
