"""Analyze paired closed-loop action-step ablations."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class EpisodeResult:
    """Metrics for one closed-loop evaluation episode."""

    seed: int
    success: bool
    steps: int
    termination_reason: str
    clipped_step_fraction: float
    clipped_element_fraction: float
    clipped_delta_x_fraction: float | None
    clipped_delta_y_fraction: float | None
    clipped_delta_z_fraction: float | None
    clipped_gripper_fraction: float | None
    maximum_action_excess: float
    mean_inference_ms: float


@dataclass(frozen=True)
class EvaluationRun:
    """One action-step setting evaluated on a seed set."""

    action_steps: int
    csv_path: Path
    episodes: dict[int, EpisodeResult]


@dataclass(frozen=True)
class RunSummary:
    """Aggregate metrics for one evaluation run."""

    action_steps: int
    successes: int
    episodes: int
    success_rate: float
    confidence_lower: float
    confidence_upper: float
    mean_steps: float
    mean_requests: float
    mean_estimated_inference_ms: float


@dataclass(frozen=True)
class PairwiseResult:
    """Paired binary comparison between two runs."""

    left_action_steps: int
    right_action_steps: int
    both_success: int
    both_failure: int
    left_only: int
    right_only: int
    exact_p_value: float


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"invalid Boolean value: {value!r}")


def _optional_float(
    row: dict[str, str],
    key: str,
) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def load_evaluation_run(
    csv_path: Path,
    action_steps: int,
) -> EvaluationRun:
    """Load and validate one closed-loop result CSV."""
    if action_steps <= 0:
        raise ValueError("action_steps must be positive")
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    with csv_path.open(
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"evaluation CSV is empty: {csv_path}")

    episodes: dict[int, EpisodeResult] = {}
    for row in rows:
        seed = int(row["seed"])
        if seed in episodes:
            raise ValueError(
                f"duplicate seed {seed} in {csv_path}"
            )

        episodes[seed] = EpisodeResult(
            seed=seed,
            success=_parse_bool(row["success"]),
            steps=int(row["steps"]),
            termination_reason=row["termination_reason"],
            clipped_step_fraction=float(
                row["clipped_step_fraction"]
            ),
            clipped_element_fraction=float(
                row["clipped_element_fraction"]
            ),
            clipped_delta_x_fraction=_optional_float(
                row,
                "clipped_delta_x_fraction",
            ),
            clipped_delta_y_fraction=_optional_float(
                row,
                "clipped_delta_y_fraction",
            ),
            clipped_delta_z_fraction=_optional_float(
                row,
                "clipped_delta_z_fraction",
            ),
            clipped_gripper_fraction=_optional_float(
                row,
                "clipped_gripper_fraction",
            ),
            maximum_action_excess=float(
                row["maximum_action_excess"]
            ),
            mean_inference_ms=float(row["mean_inference_ms"]),
        )

    return EvaluationRun(
        action_steps=action_steps,
        csv_path=csv_path,
        episodes=episodes,
    )


def validate_paired_runs(
    runs: Sequence[EvaluationRun],
) -> list[int]:
    """Require all runs to contain exactly the same seeds."""
    if len(runs) < 2:
        raise ValueError("at least two runs are required")

    expected = set(runs[0].episodes)
    for run in runs[1:]:
        actual = set(run.episodes)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                "seed mismatch for "
                f"n_action_steps={run.action_steps}: "
                f"missing={missing}, extra={extra}"
            )

    return sorted(expected)


def wilson_interval(
    successes: int,
    total: int,
    z_score: float = 1.96,
) -> tuple[float, float]:
    """Return a Wilson score confidence interval."""
    if total <= 0:
        raise ValueError("total must be positive")
    if not 0 <= successes <= total:
        raise ValueError("successes must be between zero and total")

    proportion = successes / total
    denominator = 1 + z_score**2 / total
    center = (
        proportion + z_score**2 / (2 * total)
    ) / denominator
    margin = (
        z_score
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z_score**2 / (4 * total**2)
        )
        / denominator
    )
    return center - margin, center + margin


def exact_mcnemar_p_value(
    left_only: int,
    right_only: int,
) -> float:
    """Return the two-sided exact McNemar binomial p-value."""
    if left_only < 0 or right_only < 0:
        raise ValueError("discordant counts must be non-negative")

    discordant = left_only + right_only
    if discordant == 0:
        return 1.0

    smaller = min(left_only, right_only)
    tail_probability = sum(
        math.comb(discordant, index)
        for index in range(smaller + 1)
    ) / 2**discordant
    return min(1.0, 2 * tail_probability)


def summarize_run(run: EvaluationRun) -> RunSummary:
    """Compute aggregate performance and inference metrics."""
    episodes = list(run.episodes.values())
    count = len(episodes)
    successes = sum(episode.success for episode in episodes)
    lower, upper = wilson_interval(successes, count)

    requests = [
        math.ceil(episode.steps / run.action_steps)
        for episode in episodes
    ]
    estimated_inference_ms = [
        request_count * episode.mean_inference_ms
        for request_count, episode in zip(requests, episodes)
    ]

    return RunSummary(
        action_steps=run.action_steps,
        successes=successes,
        episodes=count,
        success_rate=successes / count,
        confidence_lower=lower,
        confidence_upper=upper,
        mean_steps=sum(
            episode.steps for episode in episodes
        ) / count,
        mean_requests=sum(requests) / count,
        mean_estimated_inference_ms=(
            sum(estimated_inference_ms) / count
        ),
    )


def compare_runs(
    left: EvaluationRun,
    right: EvaluationRun,
) -> PairwiseResult:
    """Compare success outcomes on matching seeds."""
    seeds = validate_paired_runs([left, right])

    both_success = 0
    both_failure = 0
    left_only = 0
    right_only = 0

    for seed in seeds:
        left_success = left.episodes[seed].success
        right_success = right.episodes[seed].success

        if left_success and right_success:
            both_success += 1
        elif not left_success and not right_success:
            both_failure += 1
        elif left_success:
            left_only += 1
        else:
            right_only += 1

    return PairwiseResult(
        left_action_steps=left.action_steps,
        right_action_steps=right.action_steps,
        both_success=both_success,
        both_failure=both_failure,
        left_only=left_only,
        right_only=right_only,
        exact_p_value=exact_mcnemar_p_value(
            left_only,
            right_only,
        ),
    )

def render_markdown(
    runs: Sequence[EvaluationRun],
) -> str:
    """Render a reproducible Markdown ablation report."""
    seeds = validate_paired_runs(runs)
    ordered = sorted(
        runs,
        key=lambda run: run.action_steps,
    )

    lines = [
        "# ACT Action-Chunk Execution Ablation",
        "",
        f"Paired evaluation on {len(seeds)} seeds "
        f"({seeds[0]}–{seeds[-1]}).",
        "",
        "## Main results",
        "",
        "| Action steps | Success | 95% Wilson CI | "
        "Mean steps | Requests/episode | "
        "Estimated inference ms/episode |",
        "|---:|---:|---:|---:|---:|---:|",
    ]

    for run in ordered:
        summary = summarize_run(run)
        lines.append(
            f"| {summary.action_steps} "
            f"| {summary.successes}/{summary.episodes} "
            f"({summary.success_rate:.1%}) "
            f"| [{summary.confidence_lower:.1%}, "
            f"{summary.confidence_upper:.1%}] "
            f"| {summary.mean_steps:.2f} "
            f"| {summary.mean_requests:.2f} "
            f"| {summary.mean_estimated_inference_ms:.2f} |"
        )

    lines.extend([
        "",
        "## Paired McNemar comparisons",
        "",
        "| Comparison | Both pass | Both fail | "
        "Left only | Right only | Exact p-value |",
        "|---:|---:|---:|---:|---:|---:|",
    ])

    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1:]:
            result = compare_runs(left, right)
            lines.append(
                f"| {left.action_steps} vs "
                f"{right.action_steps} "
                f"| {result.both_success} "
                f"| {result.both_failure} "
                f"| {result.left_only} "
                f"| {result.right_only} "
                f"| {result.exact_p_value:.6f} |"
            )

    baseline = max(
        ordered,
        key=lambda run: run.action_steps,
    )
    baseline_failures = [
        seed
        for seed in seeds
        if not baseline.episodes[seed].success
    ]

    lines.extend([
        "",
        "## Baseline failure crossovers",
        "",
        "Baseline failures: "
        + ", ".join(map(str, baseline_failures)),
        "",
    ])

    for run in ordered:
        if run is baseline:
            continue

        rescued = [
            seed
            for seed in baseline_failures
            if run.episodes[seed].success
        ]
        regressions = [
            seed
            for seed in seeds
            if baseline.episodes[seed].success
            and not run.episodes[seed].success
        ]
        lines.extend([
            f"- `{run.action_steps}` steps rescued: "
            + ", ".join(map(str, rescued)),
            f"- `{run.action_steps}` steps regressed: "
            + ", ".join(map(str, regressions)),
        ])

    common_failures = [
        seed
        for seed in seeds
        if all(
            not run.episodes[seed].success
            for run in ordered
        )
    ]
    common_failure_text = (
        ", ".join(map(str, common_failures))
        if common_failures
        else "none"
    )
    lines.extend([
        "",
        "Failed under every setting: "
        + common_failure_text,
        "",
    ])
    return "\n".join(lines)

def _parse_run_argument(
    value: str,
) -> tuple[int, Path]:
    try:
        action_steps_text, path_text = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "run must use ACTION_STEPS=CSV_PATH"
        ) from error

    try:
        action_steps = int(action_steps_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "ACTION_STEPS must be an integer"
        ) from error

    return action_steps, Path(path_text)


def main() -> None:
    """Run paired analysis from the command line."""
    parser = argparse.ArgumentParser(
        description="Analyze paired action-step evaluations.",
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=_parse_run_argument,
        help="Evaluation in ACTION_STEPS=CSV_PATH form.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional Markdown output path.",
    )
    arguments = parser.parse_args()

    runs = [
        load_evaluation_run(path, action_steps)
        for action_steps, path in arguments.run
    ]
    report = render_markdown(runs)

    if arguments.output is not None:
        arguments.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        arguments.output.write_text(
            report,
            encoding="utf-8",
        )

    print(report)


if __name__ == "__main__":
    main()
