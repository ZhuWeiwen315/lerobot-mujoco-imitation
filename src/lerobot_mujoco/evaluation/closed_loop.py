"""Evaluate a remote ACT policy in Panda Lift closed loop."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from lerobot_mujoco.envs import PandaLiftConfig, PandaLiftEnv
from lerobot_mujoco.inference import PolicyClient


CSV_FIELDS = [
    "seed",
    "initial_cube_x",
    "initial_cube_y",
    "initial_cube_z",
    "final_cube_x",
    "final_cube_y",
    "final_cube_z",
    "success",
    "steps",
    "termination_reason",
    "maximum_cube_height",
    "clipped_steps",
    "clipped_step_fraction",
    "clipped_elements",
    "clipped_element_fraction",
    "clipped_delta_x_fraction",
    "clipped_delta_y_fraction",
    "clipped_delta_z_fraction",
    "clipped_gripper_fraction",
    "maximum_action_excess",
    "mean_inference_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ACT in closed-loop Panda Lift episodes."
    )
    parser.add_argument(
        "--socket-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--record-episodes",
        type=int,
        default=0,
        help="Record the first N evaluated episodes.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=300,
    )
    return parser.parse_args()


def create_video_writer(
    path: Path,
    fps: int,
) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (384, 384),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")

    return writer


def write_video_frame(
    writer: cv2.VideoWriter,
    image: np.ndarray,
    *,
    seed: int,
    step: int,
    action: np.ndarray,
    cube_height: float,
) -> None:
    frame = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR,
    )
    frame = cv2.resize(
        frame,
        (384, 384),
        interpolation=cv2.INTER_NEAREST,
    )

    lines = [
        f"seed={seed} step={step}",
        (
            "action="
            + np.array2string(
                action,
                precision=2,
                suppress_small=True,
            )
        ),
        f"cube_z={cube_height:.4f}",
    ]

    for line_index, line in enumerate(lines):
        y = 24 + 24 * line_index
        cv2.putText(
            frame,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    writer.write(frame)


def evaluate_episode(
    env: PandaLiftEnv,
    client: PolicyClient,
    *,
    seed: int,
    video_path: Path | None,
) -> dict[str, object]:
    observation, info = env.reset(seed=seed)
    client.reset()

    initial_cube_position = np.asarray(
        info["cube_pos"],
        dtype=np.float64,
    ).copy()
    maximum_cube_height = float(initial_cube_position[2])
    clipped_steps = 0
    clipped_elements = 0
    clipped_counts = np.zeros(4, dtype=np.int64)
    maximum_action_excess = 0.0
    inference_seconds: list[float] = []

    writer = (
        create_video_writer(
            video_path,
            env.config.control_frequency,
        )
        if video_path is not None
        else None
    )

    try:
        for step_index in range(env.config.max_episode_steps):
            image = observation["observation.images.front"]
            state = observation["observation.state"]

            inference_start = time.perf_counter()
            raw_action = client.act(image, state)
            inference_seconds.append(
                time.perf_counter() - inference_start
            )

            applied_action = np.clip(
                raw_action,
                -1.0,
                1.0,
            ).astype(np.float32)

            clipped_mask = np.abs(raw_action) > 1.0
            clipped_steps += int(clipped_mask.any())
            clipped_elements += int(clipped_mask.sum())
            clipped_counts += clipped_mask.astype(np.int64)
            maximum_action_excess = max(
                maximum_action_excess,
                float(
                    np.maximum(
                        np.abs(raw_action) - 1.0,
                        0.0,
                    ).max()
                ),
            )

            if writer is not None:
                write_video_frame(
                    writer,
                    image,
                    seed=seed,
                    step=step_index,
                    action=applied_action,
                    cube_height=float(info["cube_pos"][2]),
                )

            observation, _, terminated, truncated, info = (
                env.step(applied_action)
            )

            maximum_cube_height = max(
                maximum_cube_height,
                float(info["cube_pos"][2]),
            )

            if terminated or truncated:
                break
        else:
            raise RuntimeError(
                "Environment did not terminate or truncate within "
                f"{env.config.max_episode_steps} steps."
            )
    finally:
        if writer is not None:
            writer.release()

    total_steps = int(info["step_count"])
    total_action_elements = total_steps * 4
    final_cube_position = np.asarray(
        info["cube_pos"],
        dtype=np.float64,
    )

    return {
        "seed": seed,
        "initial_cube_x": float(initial_cube_position[0]),
        "initial_cube_y": float(initial_cube_position[1]),
        "initial_cube_z": float(initial_cube_position[2]),
        "final_cube_x": float(final_cube_position[0]),
        "final_cube_y": float(final_cube_position[1]),
        "final_cube_z": float(final_cube_position[2]),
        "success": bool(info["success"]),
        "steps": total_steps,
        "termination_reason": (
            info["termination_reason"]
            if info["termination_reason"] is not None
            else ""
        ),
        "maximum_cube_height": maximum_cube_height,
        "clipped_steps": clipped_steps,
        "clipped_step_fraction": clipped_steps / total_steps,
        "clipped_elements": clipped_elements,
        "clipped_element_fraction": (
            clipped_elements / total_action_elements
        ),
        "clipped_delta_x_fraction": (
            int(clipped_counts[0]) / total_steps
        ),
        "clipped_delta_y_fraction": (
            int(clipped_counts[1]) / total_steps
        ),
        "clipped_delta_z_fraction": (
            int(clipped_counts[2]) / total_steps
        ),
        "clipped_gripper_fraction": (
            int(clipped_counts[3]) / total_steps
        ),
        "maximum_action_excess": maximum_action_excess,
        "mean_inference_ms": (
            1000.0 * float(np.mean(inference_seconds))
        ),
    }


def main() -> None:
    args = parse_args()

    if args.num_episodes <= 0:
        raise ValueError("--num-episodes must be positive.")
    if args.record_episodes < 0:
        raise ValueError("--record-episodes must be non-negative.")
    if args.record_episodes > args.num_episodes:
        raise ValueError(
            "--record-episodes cannot exceed --num-episodes."
        )
    if args.output_csv.exists():
        raise FileExistsError(
            f"Output CSV already exists: {args.output_csv}"
        )

    args.output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.record_episodes > 0:
        if args.video_dir is None:
            raise ValueError(
                "--video-dir is required when recording episodes."
            )
        args.video_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    env = PandaLiftEnv(
        PandaLiftConfig(
            max_episode_steps=args.max_episode_steps,
        )
    )

    results: list[dict[str, object]] = []

    try:
        with PolicyClient(args.socket_path) as client:
            with args.output_csv.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=CSV_FIELDS,
                )
                writer.writeheader()
                csv_file.flush()

                for episode_index in range(args.num_episodes):
                    seed = args.seed_start + episode_index
                    video_path = (
                        args.video_dir
                        / f"seed_{seed:06d}.mp4"
                        if episode_index < args.record_episodes
                        else None
                    )

                    result = evaluate_episode(
                        env,
                        client,
                        seed=seed,
                        video_path=video_path,
                    )
                    results.append(result)
                    writer.writerow(result)
                    csv_file.flush()

                    print(
                        f"seed={seed:06d} "
                        f"success={result['success']} "
                        f"steps={result['steps']} "
                        f"reason={result['termination_reason']} "
                        f"max_z={result['maximum_cube_height']:.4f} "
                        f"clip={result['clipped_step_fraction']:.1%} "
                        f"latency={result['mean_inference_ms']:.2f}ms",
                        flush=True,
                    )
    finally:
        env.close()

    successes = sum(
        int(result["success"])
        for result in results
    )

    print("\n=== Closed-loop evaluation summary ===")
    print("episodes:", len(results))
    print("successes:", successes)
    print(
        "success rate:",
        f"{successes / len(results):.1%}",
    )
    print(
        "mean steps:",
        round(
            float(
                np.mean(
                    [result["steps"] for result in results]
                )
            ),
            2,
        ),
    )
    print(
        "mean clipped-step fraction:",
        f"{np.mean([result['clipped_step_fraction'] for result in results]):.1%}",
    )
    print(
        "mean inference latency:",
        (
            f"{np.mean([result['mean_inference_ms'] for result in results]):.2f} ms"
        ),
    )
    print("csv:", args.output_csv)


if __name__ == "__main__":
    main()
