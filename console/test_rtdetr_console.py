import unittest
from pathlib import Path

from console import rtdetr_console as app


class ConsoleTests(unittest.TestCase):
    def test_profiles_exist(self):
        self.assertIn("v1", app.VERSION_PROFILES)
        self.assertIn("v4", app.VERSION_PROFILES)

    def test_v4_train_command_uses_update_overrides(self):
        profile = app.VERSION_PROFILES["v4"]
        command, _ = app._build_train_command(profile, {
            "version": "v4",
            "config_path": str(profile.train_config),
            "output_dir": str(profile.output_dir),
            "seed": 0,
            "amp": True,
            "train_batch_size": 32,
            "val_batch_size": 64,
            "train_num_workers": 4,
            "val_num_workers": 4,
        }, "train")
        joined = " ".join(command)
        self.assertIn("train.py", joined)
        self.assertIn("--use-amp", joined)
        self.assertIn("--output-dir", joined)
        self.assertIn("train_dataloader.total_batch_size=32", joined)
        self.assertIn("val_dataloader.total_batch_size=64", joined)
        self.assertIn("checkpoint_freq=1", joined)
        self.assertEqual(command.count("-u"), 1)

    def test_v4_defaults_use_official_training_parameters(self):
        profile = app.VERSION_PROFILES["v4"]
        defaults = app._train_defaults(profile)
        self.assertEqual(defaults["weight_path"], "")
        self.assertTrue(defaults["amp"])
        self.assertEqual(defaults["seed"], 0)
        self.assertEqual(defaults["train_batch_size"], 32)
        self.assertEqual(defaults["val_batch_size"], 64)
        self.assertEqual(defaults["train_num_workers"], 4)
        self.assertEqual(defaults["val_num_workers"], 4)

    def test_v4_default_train_command_does_not_tune_from_checkpoint(self):
        profile = app.VERSION_PROFILES["v4"]
        command, _ = app._build_train_command(profile, {
            "version": "v4",
            "config_path": str(profile.train_config),
            "output_dir": str(profile.output_dir),
        }, "train")
        joined = " ".join(command)
        self.assertIn("--seed 0", joined)
        self.assertIn("--use-amp", joined)
        self.assertIn("train_dataloader.total_batch_size=32", joined)
        self.assertNotIn(" -t ", joined)
        self.assertNotIn("checkpoint/RTv4-S-hgnet.pth", joined)

    def test_v4_eval_command_uses_verified_checkpoint_by_default(self):
        profile = app.VERSION_PROFILES["v4"]
        command, _ = app._build_train_command(profile, {
            "version": "v4",
            "config_path": str(profile.train_config),
            "output_dir": str(profile.output_dir),
        }, "eval")
        joined = " ".join(command).replace("\\", "/")
        self.assertIn("-r checkpoint/RTv4-S-hgnet.pth", joined)
        self.assertIn("--test-only", joined)

    def test_v4_subprocess_env_disables_libuv_on_windows(self):
        profile = app.VERSION_PROFILES["v4"]
        env = app._subprocess_env(profile)
        self.assertEqual(env["USE_LIBUV"], "0")

    def test_v4_profile_has_official_log_default(self):
        profile = app.VERSION_PROFILES["v4"]
        self.assertIsNotNone(profile.official_log)
        self.assertIn("RTv4-S-hgnet.log", str(profile.official_log).replace("\\", "/"))

    def test_html_uses_explicit_dom_lookup_for_controls(self):
        html = app._html()
        self.assertIn("$('versionSelect').innerHTML", html)
        self.assertIn("$('trainTabBtn').onclick", html)
        self.assertIn("$('inferRunBtn').onclick", html)
        self.assertNotIn("versionSelect.innerHTML", html)
        self.assertNotIn("trainStartBtn.onclick", html)

    def test_html_log_join_escapes_newline_for_javascript(self):
        html = app._html()
        self.assertIn(r"join('\n')", html)
        self.assertNotIn("join('\n')", html)

    def test_html_command_preview_is_safe_before_status_loads(self):
        html = app._html()
        self.assertIn("if (!state) {", html)
        self.assertIn("$('trainCommandPreview').textContent = '';", html)

    def test_html_version_change_resets_version_specific_fields(self):
        html = app._html()
        self.assertIn("function resetVersionFields()", html)
        self.assertIn("$('versionSelect').onchange = () => { resetVersionFields();", html)

    def test_html_has_inline_favicon_to_avoid_browser_404(self):
        html = app._html()
        self.assertIn('<link rel="icon" href="data:,">', html)

    def test_html_log_controls_clear_and_refresh(self):
        html = app._html()
        for control_id in (
            "trainLogClearBtn",
            "trainLogRefreshBtn",
            "trainErrLogClearBtn",
            "trainErrLogRefreshBtn",
            "inferLogClearBtn",
            "inferLogRefreshBtn",
            "inferErrLogClearBtn",
            "inferErrLogRefreshBtn",
        ):
            self.assertIn(f'id="{control_id}"', html)
        self.assertIn("function clearLogView(logId)", html)
        self.assertIn("function refreshLogView(logId)", html)
        self.assertIn("logViewCleared = { trainLog:false", html)
        self.assertIn("$('trainLogClearBtn').onclick", html)

    def test_html_contains_official_compare_controls(self):
        html = app._html()
        for control_id in (
            "officialCompareSection",
            "officialLogPath",
            "officialCompareTable",
            "officialCompareRefreshBtn",
            "officialCompareDefaultBtn",
        ):
            self.assertIn(f'id=\"{control_id}\"', html)
        self.assertIn("function renderOfficialCompare()", html)

    def test_status_includes_official_compare_for_v4(self):
        profile = app.VERSION_PROFILES["v4"]
        tmp_output = Path(__file__).resolve().parent / "_tmp_console_test_output"
        tmp_output.mkdir(parents=True, exist_ok=True)
        # 控制台不会删除目录；这里创建一个专用输出目录用于测试。
        tmp_output.mkdir(parents=True, exist_ok=True)
        body = {
            "version": "v4",
            "mode": "train",
            "output_dir": str(tmp_output),
            "config_path": str(profile.train_config),
        }
        state = app._status(profile, body, "train", "train")
        self.assertIn("official_compare", state)
        self.assertIn("official_log_default", state)
        self.assertIsInstance(state["official_compare"], dict)

    def test_official_compare_falls_back_to_epoch_ap_when_best_stat_missing(self):
        profile = app.VERSION_PROFILES["v4"]
        tmp_output = Path(__file__).resolve().parent / "_tmp_console_test_output"
        tmp_output.mkdir(parents=True, exist_ok=True)

        official_log = tmp_output / "official_v4_minimal.log"
        official_log.write_text(
            "\n".join([
                "Epoch: [4] Total time: 00:10:00 (0:00:00) ",
                "best_stat: {'epoch': 4, 'coco_eval_bbox': 0.500}",
                "Epoch: [5] Total time: 00:11:00 (0:00:00) ",
                "Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.510",
                "Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.700",
            ]) + "\n",
            encoding="utf-8",
        )

        compare = app.build_official_compare(
            profile,
            {"official_log_path": str(official_log)},
            [{"epoch": 5, "ap": 0.499, "train_loss": 12.3}],
            {},
        )
        self.assertTrue(compare["ok"])
        self.assertEqual(len(compare["rows"]), 1)
        row = compare["rows"][0]
        self.assertAlmostEqual(row["official_ap_best"], 0.510, places=6)
        self.assertAlmostEqual(row["official_ap"], 0.510, places=6)
        self.assertEqual(row["official_ap_source"], "epoch_ap")

    def test_v4_detection_command_uses_verified_checkpoint_by_default(self):
        profile = app.VERSION_PROFILES["v4"]
        command = app._build_infer_command(profile, {
            "version": "v4",
            "config_path": str(profile.train_config),
            "input_path": str(Path("sample.jpg")),
            "device": "cpu",
        })
        joined = " ".join(command).replace("\\", "/")
        self.assertIn("-r checkpoint/RTv4-S-hgnet.pth", joined)

    def test_v4_preflight_accepts_verified_resources(self):
        profile = app.VERSION_PROFILES["v4"]
        body = {
            "version": "v4",
            "mode": "train",
            "config_path": str(profile.train_config),
            "weight_path": str(profile.train_weight),
            "output_dir": str(profile.output_dir),
        }
        preflight = app._run_preflight(profile, body, "eval", "train")
        failed = [item["name"] for item in preflight["checks"] if not item["ok"]]
        # 训练可能正在后台运行，这个检查在开发机上不应作为硬失败。
        failed = [name for name in failed if name != "train process not already running"]
        self.assertEqual(failed, [])

    def test_v4_resume_command_uses_last_checkpoint(self):
        profile = app.VERSION_PROFILES["v4"]
        command, _ = app._build_train_command(profile, {
            "version": "v4",
            "config_path": str(profile.train_config),
            "output_dir": str(profile.output_dir),
        }, "resume")
        joined = " ".join(command)
        self.assertIn("last.pth", joined)

    def test_v1_detection_command_uses_image_flag(self):
        profile = app.VERSION_PROFILES["v1"]
        command = app._build_infer_command(profile, {
            "version": "v1",
            "config_path": str(profile.train_config),
            "weight_path": str(profile.output_dir / "checkpoint.pth"),
            "input_path": str(Path("sample.jpg")),
            "device": "cpu",
        })
        joined = " ".join(command)
        self.assertIn("infer.py", joined)
        self.assertIn("-f", joined)
        self.assertIn("-d cpu", joined)


if __name__ == "__main__":
    unittest.main()
