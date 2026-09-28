"""Dataset collection and conversion utilities."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .lerobot_converter import (
        ConversionSummary,
        convert_raw_episodes,
        make_lerobot_features,
        validate_raw_episode,
    )
    from .raw_episode import collect_raw_episode, save_raw_episode

__all__ = [
    "ConversionSummary",
    "collect_raw_episode",
    "convert_raw_episodes",
    "make_lerobot_features",
    "save_raw_episode",
    "validate_raw_episode",
]


def __getattr__(name: str):
    """Load simulation and training dependencies only when requested."""
    if name in {"collect_raw_episode", "save_raw_episode"}:
        from .raw_episode import collect_raw_episode, save_raw_episode

        return {
            "collect_raw_episode": collect_raw_episode,
            "save_raw_episode": save_raw_episode,
        }[name]

    if name in {
        "ConversionSummary",
        "convert_raw_episodes",
        "make_lerobot_features",
        "validate_raw_episode",
    }:
        from .lerobot_converter import (
            ConversionSummary,
            convert_raw_episodes,
            make_lerobot_features,
            validate_raw_episode,
        )

        return {
            "ConversionSummary": ConversionSummary,
            "convert_raw_episodes": convert_raw_episodes,
            "make_lerobot_features": make_lerobot_features,
            "validate_raw_episode": validate_raw_episode,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
