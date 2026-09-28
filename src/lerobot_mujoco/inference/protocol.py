"""Unix-socket protocol shared by simulation and training environments."""

from __future__ import annotations

import socket
import struct
from pathlib import Path

import numpy as np


IMAGE_SHAPE = (96, 96, 3)
STATE_SHAPE = (12,)
ACTION_SHAPE = (4,)

IMAGE_BYTES = int(np.prod(IMAGE_SHAPE))
STATE_BYTES = int(np.prod(STATE_SHAPE)) * np.dtype("<f4").itemsize
ACTION_BYTES = int(np.prod(ACTION_SHAPE)) * np.dtype("<f4").itemsize

COMMAND_ACT = b"A"
COMMAND_RESET = b"R"
COMMAND_QUIT = b"Q"

STATUS_OK = b"K"
STATUS_ERROR = b"E"


def receive_exact(connection: socket.socket, size: int) -> bytes:
    """Receive exactly size bytes or raise if the peer disconnects."""

    chunks = bytearray()

    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError(
                f"Socket closed after {len(chunks)} of {size} bytes."
            )
        chunks.extend(chunk)

    return bytes(chunks)


def send_error(connection: socket.socket, message: str) -> None:
    """Send a UTF-8 error response."""

    payload = message.encode("utf-8", errors="replace")
    connection.sendall(
        STATUS_ERROR
        + struct.pack("!I", len(payload))
        + payload
    )


class PolicyClient:
    """Persistent client used by the MuJoCo simulation process."""

    def __init__(
        self,
        socket_path: str | Path,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.timeout_seconds = float(timeout_seconds)
        self._socket: socket.socket | None = None

    def connect(self) -> None:
        if self._socket is not None:
            raise RuntimeError("PolicyClient is already connected.")

        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(self.timeout_seconds)
        connection.connect(str(self.socket_path))
        self._socket = connection

    def reset(self) -> None:
        """Clear ACT's queued actions for a new environment episode."""

        connection = self._require_connection()
        connection.sendall(COMMAND_RESET)
        self._receive_response(expected_payload_bytes=0)

    def act(
        self,
        image: np.ndarray,
        state: np.ndarray,
    ) -> np.ndarray:
        """Return one un-clipped environment-space action."""

        image = np.asarray(image)
        state = np.asarray(state)

        if image.shape != IMAGE_SHAPE or image.dtype != np.uint8:
            raise ValueError(
                "Expected image with "
                f"shape={IMAGE_SHAPE}, dtype=uint8; "
                f"received shape={image.shape}, dtype={image.dtype}."
            )

        if state.shape != STATE_SHAPE:
            raise ValueError(
                f"Expected state shape={STATE_SHAPE}; "
                f"received shape={state.shape}."
            )

        image_bytes = np.ascontiguousarray(image).tobytes()
        state_bytes = np.ascontiguousarray(
            state,
            dtype="<f4",
        ).tobytes()

        connection = self._require_connection()
        connection.sendall(
            COMMAND_ACT + image_bytes + state_bytes
        )

        payload = self._receive_response(
            expected_payload_bytes=ACTION_BYTES
        )
        return (
            np.frombuffer(payload, dtype="<f4")
            .copy()
            .reshape(ACTION_SHAPE)
        )

    def close(self) -> None:
        if self._socket is None:
            return

        try:
            self._socket.sendall(COMMAND_QUIT)
            self._receive_response(expected_payload_bytes=0)
        except (ConnectionError, BrokenPipeError, OSError):
            pass
        finally:
            self._socket.close()
            self._socket = None

    def _require_connection(self) -> socket.socket:
        if self._socket is None:
            raise RuntimeError(
                "PolicyClient is not connected. Call connect() first."
            )
        return self._socket

    def _receive_response(
        self,
        expected_payload_bytes: int,
    ) -> bytes:
        connection = self._require_connection()
        status = receive_exact(connection, 1)

        if status == STATUS_OK:
            return receive_exact(
                connection,
                expected_payload_bytes,
            )

        if status == STATUS_ERROR:
            message_size = struct.unpack(
                "!I",
                receive_exact(connection, 4),
            )[0]
            message = receive_exact(
                connection,
                message_size,
            ).decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Policy server error: {message}"
            )

        raise RuntimeError(
            f"Unknown policy-server response: {status!r}"
        )

    def __enter__(self) -> "PolicyClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
