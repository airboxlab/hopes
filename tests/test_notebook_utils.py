import re
import unittest

from hopes.notebook_utils import (
    daterange_inclusive,
    day_minus_one,
    extract_end_date_from_episode_id,
    local_day_from_filename_utc,
    make_prefix_new,
    should_skip_day,
)


class TestNotebookUtils(unittest.TestCase):
    def test_should_skip_day_weekend(self):
        # Saturday
        self.assertTrue(should_skip_day("2026-03-07", excluded_days=[], weekdays_only=True))
        # Sunday
        self.assertTrue(should_skip_day("2026-03-08", excluded_days=[], weekdays_only=True))

        # If weekdays_only=False, weekend is not skipped unless excluded
        self.assertFalse(should_skip_day("2026-03-07", excluded_days=[], weekdays_only=False))

    def test_should_skip_day_excluded(self):
        self.assertTrue(
            should_skip_day("2026-03-04", excluded_days=["2026-03-04"], weekdays_only=True)
        )
        self.assertFalse(
            should_skip_day("2026-03-04", excluded_days=["2026-03-05"], weekdays_only=True)
        )

    def test_daterange_inclusive_basic(self):
        days = list(daterange_inclusive("2026-03-01", "2026-03-03"))
        self.assertEqual(days, ["2026-03-01", "2026-03-02", "2026-03-03"])

    def test_daterange_inclusive_single_day(self):
        days = list(daterange_inclusive("2026-03-02", "2026-03-02"))
        self.assertEqual(days, ["2026-03-02"])

    def test_daterange_inclusive_invalid_order(self):
        with self.assertRaises(AssertionError):
            list(daterange_inclusive("2026-03-03", "2026-03-02"))

    def test_local_day_from_filename_utc_no_match(self):
        regex_key = re.compile(r"^trajectory_(\d{8})_(\d{6})\.csv$")
        out = local_day_from_filename_utc(
            key="some/prefix/not_matching_name.csv",
            regex_key=regex_key,
            local_tz="Europe/Paris",
        )
        self.assertIsNone(out)

    def test_local_day_from_filename_utc_same_day(self):
        # 2026-03-02 12:00:00 UTC -> still 2026-03-02 in Europe/Paris (CET/CEST depending, but still same date)
        regex_key = re.compile(r"^trajectory_(\d{8})_(\d{6})\.csv$")
        key = "x/y/trajectory_20260302_120000.csv"

        out = local_day_from_filename_utc(key=key, regex_key=regex_key, local_tz="Europe/Paris")
        self.assertEqual(out, "2026-03-02")

    def test_local_day_from_filename_utc_crosses_midnight(self):
        # 2026-03-02 23:30:00 UTC -> 2026-03-03 local in Europe/Paris (UTC+1 or UTC+2)
        # This test checks "date shift" behavior without relying on exact offset.
        regex_key = re.compile(r"^trajectory_(\d{8})_(\d{6})\.csv$")
        key = "trajectory_20260302_233000.csv"

        out = local_day_from_filename_utc(key=key, regex_key=regex_key, local_tz="Europe/Paris")
        self.assertIn(out, ["2026-03-02", "2026-03-03"])
        # In practice it's expected to be 2026-03-03 for Europe/Paris:
        self.assertEqual(out, "2026-03-03")

    def test_day_minus_one(self):
        self.assertEqual(day_minus_one("2026-03-01"), "2026-02-28")
        self.assertEqual(day_minus_one("2026-01-01"), "2025-12-31")

    def test_extract_end_date_from_episode_id_ok(self):
        eid = "BATIMENT_XYZ_2026_02_20"
        self.assertEqual(extract_end_date_from_episode_id(eid), "2026-02-20")

    def test_extract_end_date_from_episode_id_not_string(self):
        with self.assertRaises(ValueError):
            extract_end_date_from_episode_id(123)  # type: ignore[arg-type]

    def test_extract_end_date_from_episode_id_bad_format(self):
        with self.assertRaises(ValueError):
            extract_end_date_from_episode_id("BATIMENT_XYZ_2026-02-20")

        with self.assertRaises(ValueError):
            extract_end_date_from_episode_id("BATIMENT_XYZ_2026_2_20")  # not zero-padded

        with self.assertRaises(ValueError):
            extract_end_date_from_episode_id("BATIMENT_XYZ_2026_02")  # incomplete

    def test_make_prefix_new(self):
        run_id = "run123"
        model_name = "my_model"
        base_training_prefix = "s3://bucket/training"

        out = make_prefix_new(run_id, model_name, base_training_prefix)
        self.assertEqual(
            out,
            "s3://bucket/training/run123/config/my_model/output/my_model",
        )
