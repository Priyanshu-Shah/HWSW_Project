"""Find and load board files from a boards directory (default: library/boards)."""

from pathlib import Path

from hwsw.json_files import read_json
from hwsw.platforms.board_model import Board
from hwsw.repo_paths import LIBRARY_BOARDS_DIR


class BoardNotFoundError(LookupError):
    def __init__(self, board_id: str, boards_dir: Path):
        self.board_id = board_id
        self.available = list_board_ids(boards_dir)
        super().__init__(
            f"unknown platform '{board_id}'; boards available in {boards_dir}: "
            f"{', '.join(self.available) or 'none'}"
        )


def list_board_ids(boards_dir: Path = LIBRARY_BOARDS_DIR) -> list[str]:
    if not boards_dir.is_dir():
        return []
    return sorted(path.stem for path in boards_dir.glob("*.json"))


def load_board(board_id: str, boards_dir: Path = LIBRARY_BOARDS_DIR) -> Board:
    path = boards_dir / f"{board_id}.json"
    if not path.is_file():
        raise BoardNotFoundError(board_id, boards_dir)
    return Board.model_validate(read_json(path))
