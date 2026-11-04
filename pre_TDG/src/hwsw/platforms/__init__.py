"""Board specifications, read from the library team's files."""

from hwsw.platforms.board_loader import BoardNotFoundError, list_board_ids, load_board
from hwsw.platforms.board_model import RESOURCE_KEYS, Board

__all__ = ["RESOURCE_KEYS", "Board", "BoardNotFoundError", "list_board_ids", "load_board"]
