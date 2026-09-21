import contextlib
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        with patch.dict(os.environ, {"CHECKPOINT_DIR": self.temp.name, "NASA_FIRMS_MAP_KEY": "test-key-not-a-credential", "DATABASE_URL": ""}):
            spec = importlib.util.spec_from_file_location("collector_test", ROOT / "dashboard/live_ingest.py")
            self.collector = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.collector)

    def observations(self):
        return pd.DataFrame({
            "acq_date": ["2026-09-21"] * 3 + ["2026-09-20"],
            "acq_time": [41, 1109, 1200, 1200],
            "confidence": ["n", "n", "h", "h"],
            "frp": [3.0, 12.0, 25.0, 30.0],
            "latitude": [46.0, 46.1, 46.2, 46.2],
            "longitude": [32.0, 32.1, 32.2, 32.2],
            "region": ["south_ukraine_black_sea_box"] * 4,
            "firms_source": ["VIIRS_NOAA20_NRT"] * 4,
            "ingested_at_utc": [pd.Timestamp("2026-09-21T13:00Z")] * 4,
        })

    def test_repeat_pull_deduplicates_and_filters_observation_day(self):
        with patch.object(self.collector, "fetch_region_source", return_value=self.observations()):
            first = self.collector.run_once("2026-09-21")
            second = self.collector.run_once("2026-09-21")
        self.assertEqual(first["records_returned"], 3)
        self.assertEqual(first["new_detections"], 3)
        self.assertEqual(first["new_review_alerts"], 2)
        self.assertEqual(second["records_returned"], 3)
        self.assertEqual(second["new_detections"], 0)
        self.assertEqual(second["new_review_alerts"], 0)
        saved = pd.read_csv(Path(self.temp.name) / "live_firms_detections.csv")
        self.assertEqual(saved.shape[0], 3)
        self.assertEqual(set(saved["alert_priority"]), {"monitor", "medium_review", "high_review"})
        self.assertTrue(pd.to_datetime(saved["timestamp"], utc=True).dt.strftime("%Y-%m-%d").eq("2026-09-21").all())

    def test_api_failure_redacts_key(self):
        message = "failed URL /csv/" + self.collector.MAP_KEY + "/source"
        with patch.object(self.collector, "fetch_region_source", side_effect=RuntimeError(message)):
            with self.assertRaises(RuntimeError) as caught:
                self.collector.run_once("2026-09-21")
        self.assertNotIn(self.collector.MAP_KEY, str(caught.exception))
        self.assertIn("[REDACTED]", str(caught.exception))

    def test_missing_key_fails_before_network(self):
        self.collector.MAP_KEY = ""
        with patch.object(self.collector, "fetch_region_source") as fetch:
            with self.assertRaisesRegex(RuntimeError, "missing"):
                self.collector.run_once("2026-09-21")
            fetch.assert_not_called()


class DashboardTests(unittest.TestCase):
    def test_empty_install_has_no_exception_and_reports_missing_data(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict(os.environ, {"CHECKPOINT_DIR": folder, "DATABASE_URL": ""}):
                app = AppTest.from_file(str(ROOT / "dashboard/app.py"), default_timeout=30).run()
                self.assertEqual(len(app.exception), 0, str(app.exception))
                self.assertTrue(any("not available" in item.value for item in app.warning))
                self.assertTrue(any("recorded study totals" in item.value for item in app.caption))


if __name__ == "__main__":
    unittest.main()
