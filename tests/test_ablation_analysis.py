"""Tests for paired closed-loop ablation analysis."""

import csv
import tempfile
import unittest
from pathlib import Path

from lerobot_mujoco.evaluation.analyze_ablation import (
    compare_runs,
    exact_mcnemar_p_value,
    load_evaluation_run,
    render_markdown,
    summarize_run,
    validate_paired_runs,
    wilson_interval,
)


FIELDNAMES = [
    "seed",
    "success",
    "steps",
    "termination_reason",
    "clipped_step_fraction",
    "clipped_element_fraction",
    "clipped_delta_x_fraction",
    "clipped_delta_y_fraction",
    "clipped_delta_z_fraction",
    "clipped_gripper_fraction",
    "maximum_action_excess",
    "mean_inference_ms",
]


def write_csv(path: Path, outcomes: list[bool]) -> None:
    """Write a minimal valid evaluation CSV."""
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES,
        )
        writer.writeheader()

        for index, success in enumerate(outcomes):
            writer.writerow({
                "seed": 1000 + index,
                "success": str(success),
                "steps": 80 if success else 300,
                "termination_reason": (
                    "success"
                    if success
                    else "max_episode_steps"
                ),
                "clipped_step_fraction": 0.98,
                "clipped_element_fraction": 0.245,
                "clipped_delta_x_fraction": 0.0,
                "clipped_delta_y_fraction": 0.0,
                "clipped_delta_z_fraction": 0.0,
                "clipped_gripper_fraction": 0.98,
                "maximum_action_excess": 0.1,
                "mean_inference_ms": 2.5,
            })


class AblationStatisticsTest(unittest.TestCase):
    """Validate statistics used in the report."""

    def test_experiment_statistics(self) -> None:
        lower, upper = wilson_interval(44, 50)

        self.assertAlmostEqual(lower, 0.76195, places=4)
        self.assertAlmostEqual(upper, 0.94382, places=4)
        self.assertAlmostEqual(
            exact_mcnemar_p_value(4, 15),
            0.019211,
            places=6,
        )

    def test_invalid_counts_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            wilson_interval(1, 0)

        with self.assertRaises(ValueError):
            exact_mcnemar_p_value(-1, 2)


class AblationCsvTest(unittest.TestCase):
    """Validate loading, pairing, and report generation."""

    def test_summary_and_pairwise_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left_path = root / "left.csv"
            right_path = root / "right.csv"
            write_csv(
                left_path,
                [True, False, True, False],
            )
            write_csv(
                right_path,
                [True, True, False, False],
            )

            left = load_evaluation_run(left_path, 5)
            right = load_evaluation_run(right_path, 20)
            summary = summarize_run(right)
            comparison = compare_runs(left, right)

            self.assertEqual(summary.successes, 2)
            self.assertEqual(summary.mean_steps, 190.0)
            self.assertEqual(summary.mean_requests, 9.5)
            self.assertEqual(
                summary.mean_estimated_inference_ms,
                23.75,
            )
            self.assertEqual(comparison.both_success, 1)
            self.assertEqual(comparison.both_failure, 1)
            self.assertEqual(comparison.left_only, 1)
            self.assertEqual(comparison.right_only, 1)
            self.assertEqual(comparison.exact_p_value, 1.0)

    def test_seed_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            short_path = root / "short.csv"
            full_path = root / "full.csv"
            write_csv(short_path, [True])
            write_csv(full_path, [True, False])

            short = load_evaluation_run(short_path, 1)
            full = load_evaluation_run(full_path, 5)

            with self.assertRaisesRegex(
                ValueError,
                "seed mismatch",
            ):
                validate_paired_runs([short, full])

    def test_report_contains_expected_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [
                root / "one.csv",
                root / "five.csv",
                root / "twenty.csv",
            ]
            outcomes = [
                [True, False, False, True],
                [True, True, False, True],
                [True, True, True, True],
            ]
