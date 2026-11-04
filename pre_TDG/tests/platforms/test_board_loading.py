import pytest

from hwsw.platforms import BoardNotFoundError, list_board_ids, load_board


def test_board_fields(zcu104):
    assert zcu104.cpu.cores == 4
    assert zcu104.fpga.capacity["dsp"] == 1728
    assert zcu104.default_interface == "axi_hp0"
    assert zcu104.default_interface_spec.bandwidth_Bps == 2.0e9


def test_unknown_board_lists_the_available_ones(boards_dir):
    with pytest.raises(BoardNotFoundError) as unknown:
        load_board("no_such_board", boards_dir)
    assert unknown.value.available == ["kv260", "pynq_z2", "zcu104"]
    assert "kv260, pynq_z2, zcu104" in str(unknown.value)


def test_missing_boards_folder_means_no_boards(tmp_path):
    assert list_board_ids(tmp_path / "does_not_exist") == []


def test_default_interface_must_exist(boards_dir, tmp_path):
    from hwsw.json_files import read_json, write_json

    broken = read_json(boards_dir / "zcu104.json")
    broken["interfaces"]["default"] = "axi_hp9"
    write_json(tmp_path / "zcu104.json", broken)
    with pytest.raises(ValueError, match="axi_hp9"):
        load_board("zcu104", tmp_path)
