"""Where the app finds board files."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from hwsw.platforms import list_board_ids
from hwsw.repo_paths import FIXTURE_BOARDS_DIR, LIBRARY_BOARDS_DIR

BOARDS_DIR_ENV_VAR = "HWSW_BOARDS_DIR"


@dataclass(frozen=True)
class AppSettings:
    boards_dir: Path
    boards_source: str  # "library" | "custom" | "fixtures"; shown in the UI


def resolve_settings(environment: Mapping[str, str] = os.environ) -> AppSettings:
    """Board folder: $HWSW_BOARDS_DIR if set, else library/boards, else the test fixtures.

    The fixture fallback keeps the app usable until the library team delivers real board files.
    The UI says which source is in use, because fixture values are not authoritative.
    """
    if BOARDS_DIR_ENV_VAR in environment:
        return AppSettings(Path(environment[BOARDS_DIR_ENV_VAR]), "custom")
    if list_board_ids(LIBRARY_BOARDS_DIR):
        return AppSettings(LIBRARY_BOARDS_DIR, "library")
    return AppSettings(FIXTURE_BOARDS_DIR, "fixtures")
