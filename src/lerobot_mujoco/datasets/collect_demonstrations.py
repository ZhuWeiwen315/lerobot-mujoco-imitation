"""Resumable scripted demonstration collection for Panda Lift."""

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from lerobot_mujoco.datasets.raw_episode import (
    collect_raw_episode,
    save_raw_episode,
)
from lerobot_mujoco.envs import PandaLiftConfig, PandaLiftEnv
from lerobot_mujoco.experts import PandaLiftFSMExpert


MANIFEST_FIELDS = [
    "seed",
    "file",
    "success",
    "steps",
    "termination_reason",
    "status",
]


def parse_args() -> argparse.Namespace:
    """Parse demonstration collection arguments."""
    parser = argparse.ArgumentParser(
        description="Collect resumable Panda Lift expert demonstrations."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for episode_XXXXXX.npz files.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        required=True,
        help="First inclusive environment seed.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        required=True,
        help="Number of consecutive seeds to attempt.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="CSV manifest path; defaults inside output-dir.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Validate and skip existing successful episode files.",
    )
    return parser.parse_args()


def episode_path(output_dir: Path, seed: int) -> Path:
    """Return the deterministic file path for one seed."""
    return output_dir / f"episode_{seed:06d}.npz"


def read_existing_episode(
    path: Path,
    *,
    expected_seed: int,
) -> dict[str, Any]:
    """Read compact metadata from an existing successful episode."""
    with np.load(path, allow_pickle=False) as episode:
        required = {
            "images",
            "states",
            "actions",
            "seed",
            "success",
            "steps",
            "termination_reason",
        }
        missing = required.difference(episode.files)
        if missing:
            raise ValueError(
                f"Existing episode {path} is missing keys: "
                f"{sorted(missing)}"
            )

        seed = int(episode["seed"].item())
        success = bool(episode["success"].item())
        steps = int(episode["steps"].item())
        termination_reason = str(
            episode["termination_reason"].item()
        )

        lengths = {
            len(episode["images"]),
            len(episode["states"]),
            len(episode["actions"]),
        }

    if seed != expected_seed:
        raise ValueError(
            f"Episode {path} contains seed {seed}, "
            f"expected {expected_seed}."
        )
    if not success:
        raise ValueError(
            f"Existing episode is not successful: {path}"
        )
    if len(lengths) != 1 or steps not in lengths:
        raise ValueError(
            f"Existing episode arrays are not aligned: {path}"
        )

    return {
        "seed": seed,
        "file": path.name,
        "success": success,
        "steps": steps,
        "termination_reason": termination_reason,
        "status": "existing",
    }


def write_manifest(
    manifest_path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Atomically replace the CSV manifest."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path.with_suffix(
        manifest_path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=MANIFEST_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)

    temporary_path.replace(manifest_path)


def save_episode_atomically(
    episode: dict[str, Any],
    output_path: Path,
) -> Path:
    """Write an episode completely before exposing its final filename."""
    temporary_path = output_path.with_suffix(".tmp.npz")
    save_raw_episode(episode, temporary_path)
    temporary_path.replace(output_path)
    return output_path


def collect_demonstrations(
    *,
    output_dir: Path,
    seed_start: int,
    num_episodes: int,
    manifest_path: Path,
    resume: bool,
) -> list[dict[str, Any]]:
    """Collect a consecutive, auditable range of demonstrations."""
    if seed_start < 0:
        raise ValueError("seed_start must be non-negative.")
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive.")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    env = PandaLiftEnv(
        PandaLiftConfig(
            camera_name="sideview",
            image_size=96,
            control_frequency=20,
            max_episode_steps=300,
            lift_height_margin=0.08,
            success_hold_steps=10,
        )
    )
    expert = PandaLiftFSMExpert()

    try:
        for seed in range(
            seed_start,
            seed_start + num_episodes,
        ):
            output_path = episode_path(output_dir, seed)

            if output_path.exists():
                if not resume:
                    raise FileExistsError(
                        "Episode already exists; use --resume "
                        f"to validate and skip it: {output_path}"
                    )

                row = read_existing_episode(
                    output_path,
                    expected_seed=seed,
                )
                rows.append(row)
                write_manifest(manifest_path, rows)

                print(
                    f"seed={seed:06d} status=existing "
                    f"steps={row['steps']}"
                )
                continue

            episode = collect_raw_episode(
                env=env,
                expert=expert,
                seed=seed,
            )

            row = {
                "seed": seed,
                "file": "",
                "success": bool(episode["success"]),
                "steps": int(episode["steps"]),
                "termination_reason": str(
                    episode["termination_reason"]
                ),
                "status": "failed",
            }

            if episode["success"]:
                save_episode_atomically(
                    episode,
                    output_path,
                )
                row["file"] = output_path.name
                row["status"] = "collected"

            rows.append(row)
            write_manifest(manifest_path, rows)

            print(
                f"seed={seed:06d} "
                f"status={row['status']} "
                f"steps={row['steps']} "
                f"reason={row['termination_reason']}"
            )
    finally:
        env.close()

    return rows


def main() -> None:
    """Run collection and report the resulting split."""
    args = parse_args()
    manifest_path = (
        args.manifest
        if args.manifest is not None
        else args.output_dir / "manifest.csv"
    )

    rows = collect_demonstrations(
        output_dir=args.output_dir,
        seed_start=args.seed_start,
        num_episodes=args.num_episodes,
        manifest_path=manifest_path,
        resume=args.resume,
    )

    successes = sum(bool(row["success"]) for row in rows)
    failures = len(rows) - successes
    mean_steps = (
        sum(int(row["steps"]) for row in rows) / len(rows)
    )

    print("\n=== Collection summary ===")
    print("attempted:", len(rows))
    print("successful:", successes)
    print("failed:", failures)
    print("success rate:", f"{successes / len(rows):.1%}")
    print("mean steps:", round(mean_steps, 2))
    print("output directory:", args.output_dir)
    print("manifest:", manifest_path)

    if failures:
        raise RuntimeError(
            f"{failures} demonstration episodes failed."
        )


if __name__ == "__main__":
    main()
