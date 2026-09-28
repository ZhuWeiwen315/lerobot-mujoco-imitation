"""Persistent ACT policy server running in the training environment."""

from __future__ import annotations

import argparse
import socket
from pathlib import Path

import numpy as np
import torch

from lerobot_mujoco.inference.protocol import (
    ACTION_SHAPE,
    COMMAND_ACT,
    COMMAND_QUIT,
    COMMAND_RESET,
    IMAGE_BYTES,
    IMAGE_SHAPE,
    STATE_BYTES,
    STATE_SHAPE,
    STATUS_OK,
    receive_exact,
    send_error,
)
from lerobot_mujoco.training import load_act_policy_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Serve a trained ACT policy over a Unix-domain socket."
        )
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--socket-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    parser.add_argument(
        "--action-steps",
        type=int,
        default=None,
        help=(
            "Optional number of queued actions per model query. "
            "Defaults to the value saved in the checkpoint."
        ),
    )
    return parser.parse_args()


def prepare_observation(
    image: np.ndarray,
    state: np.ndarray,
) -> dict[str, torch.Tensor]:
    """Convert simulator HWC uint8 input to ACT CHW float input."""

    model_image = (
        torch.from_numpy(image.copy())
        .permute(2, 0, 1)
        .contiguous()
        .to(dtype=torch.float32)
        .div(255.0)
    )
    model_state = torch.from_numpy(
        state.copy()
    ).to(dtype=torch.float32)

    return {
        "observation.images.front": model_image,
        "observation.state": model_state,
    }


def serve_connection(
    connection: socket.socket,
    policy,
    preprocessor,
    postprocessor,
) -> None:
    """Serve one persistent simulation client."""

    while True:
        command = connection.recv(1)

        if not command:
            return

        try:
            if command == COMMAND_RESET:
                policy.reset()
                connection.sendall(STATUS_OK)
                continue

            if command == COMMAND_QUIT:
                connection.sendall(STATUS_OK)
                return

            if command != COMMAND_ACT:
                raise ValueError(
                    f"Unknown command byte: {command!r}"
                )

            image_payload = receive_exact(
                connection,
                IMAGE_BYTES,
            )
            state_payload = receive_exact(
                connection,
                STATE_BYTES,
            )

            image = (
                np.frombuffer(image_payload, dtype=np.uint8)
                .copy()
                .reshape(IMAGE_SHAPE)
            )
            state = (
                np.frombuffer(state_payload, dtype="<f4")
                .copy()
                .reshape(STATE_SHAPE)
            )

            observation = prepare_observation(image, state)
            processed = preprocessor(observation)
            normalized_action = policy.select_action(processed)
            environment_action = postprocessor(normalized_action)

            if isinstance(environment_action, torch.Tensor):
                environment_action = (
                    environment_action.detach().cpu().numpy()
                )

            action = np.asarray(
                environment_action,
                dtype="<f4",
            ).reshape(ACTION_SHAPE)

            if not np.isfinite(action).all():
                raise RuntimeError(
                    f"Policy produced non-finite action: {action}"
                )

            connection.sendall(
                STATUS_OK
                + np.ascontiguousarray(action).tobytes()
            )

        except Exception as error:
            send_error(
                connection,
                f"{type(error).__name__}: {error}",
            )


def main() -> None:
    args = parse_args()

    if args.action_steps is not None and args.action_steps <= 0:
        raise ValueError("--action-steps must be positive.")

    if args.socket_path.exists():
        raise FileExistsError(
            "Socket path already exists. Refusing to overwrite it: "
            f"{args.socket_path}"
        )

    args.socket_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    policy, preprocessor, postprocessor = (
        load_act_policy_bundle(
            args.checkpoint_dir,
            device=args.device,
        )
    )

    if args.action_steps is not None:
        if args.action_steps > policy.config.chunk_size:
            raise ValueError(
                "--action-steps cannot exceed checkpoint chunk_size="
                f"{policy.config.chunk_size}."
            )
        policy.config.n_action_steps = args.action_steps

    policy.eval()
    policy.reset()

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

    try:
        server.bind(str(args.socket_path))
        server.listen(1)

        print("=== ACT policy server ===", flush=True)
        print("checkpoint:", args.checkpoint_dir, flush=True)
        print("socket:", args.socket_path, flush=True)
        print("device:", args.device, flush=True)
        print(
            "chunk size:",
            policy.config.chunk_size,
            flush=True,
        )
        print(
            "executed action steps:",
            policy.config.n_action_steps,
            flush=True,
        )
        print("status: ready", flush=True)

        while True:
            connection, _ = server.accept()
            print("client connected", flush=True)

            try:
                serve_connection(
                    connection,
                    policy,
                    preprocessor,
                    postprocessor,
                )
            finally:
                connection.close()
                policy.reset()
                print("client disconnected", flush=True)

    finally:
        server.close()
        if args.socket_path.exists():
            args.socket_path.unlink()
        print("policy server stopped", flush=True)


if __name__ == "__main__":
    main()
