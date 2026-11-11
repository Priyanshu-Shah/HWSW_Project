"""`platform_lookup`: the board list, or one board's numbers, from the library's board files."""

import json
from pathlib import Path

from hwsw.llm.tools.tool import Tool, object_schema
from hwsw.platforms import list_board_ids, load_board


def board_list_text(boards_dir: Path) -> str:
    """One line per board: `id: summary`."""
    return "\n".join(
        f"{board_id}: {load_board(board_id, boards_dir).summary}"
        for board_id in list_board_ids(boards_dir)
    )


def platform_lookup_tool(boards_dir: Path) -> Tool:
    def platform_lookup(board_id: str = "") -> str:
        if not board_id:
            return board_list_text(boards_dir)
        board = load_board(board_id, boards_dir)  # an unknown id lists the known ones
        return json.dumps(board.model_dump(), indent=1)

    return Tool(
        name="platform_lookup",
        description=(
            "Board specifications from the project library. With no board_id: the list of "
            "boards. With a board_id: its CPU, FPGA capacity, memory and interfaces."
        ),
        parameters=object_schema({"board_id": {"type": "string"}}, required=[]),
        run=platform_lookup,
    )
