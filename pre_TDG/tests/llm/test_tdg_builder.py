"""Stage B with scripted replies that rebuild the sobel_edge fixture graph step by step."""

import copy
import json

import pytest

from fake_ollama import FakeOllama, call, text
from hwsw.conversation import UserInput
from hwsw.json_files import read_json
from hwsw.llm.pipeline_agent import PipelineAgent
from hwsw.llm.tdg_builder import TdgBuilder, TdgBuildError
from hwsw.platforms import load_board
from hwsw.repo_paths import FIXTURE_TDGS_DIR
from hwsw.validation import validate_tdg

FIXTURE = read_json(FIXTURE_TDGS_DIR / "sobel_edge.tdg.json")
SPEC = {
    "task_summary": "Sobel edge detection on 1080p video at 60 fps",
    "platform": "kv260",
    "arrival": FIXTURE["arrival"],
    "timing": {"deadline_s": 0.016667, "hardness": "hard", "derived_from": "60 fps (D3)"},
    "assumptions": [],
    "open_questions": [],
}
NODE_FIELDS = ("id", "name", "phase", "affinity", "kind")


def reply(value) -> dict:
    return text(json.dumps(value))


def nodes_reply() -> dict:
    nodes = [
        {**{key: node[key] for key in NODE_FIELDS}, "purpose": "..."} for node in FIXTURE["nodes"]
    ]
    return reply({"graph_id": "sobel_edge", "nodes": nodes})


def edges_reply(edges=None) -> dict:
    edges = edges or [{**edge, "bytes": edge["bytes"]} for edge in FIXTURE["edges"]]
    edges[0] = {**edges[0], "bytes": "1080*1920*3"}  # a formula, evaluated by the builder
    return reply({"edges": edges})


def payload_reply(node_id: str, **changes) -> dict:
    node = next(node for node in FIXTURE["nodes"] if node["id"] == node_id)
    payload = {key: value for key, value in node.items() if key not in NODE_FIELDS}
    return reply({**copy.deepcopy(payload), **changes})


def payload_replies() -> list[dict]:
    return [payload_reply(node["id"]) for node in FIXTURE["nodes"]]


@pytest.fixture
def kv260(boards_dir):
    return load_board("kv260", boards_dir)


def test_builds_the_fixture_graph(kv260):
    fake = FakeOllama(nodes_reply(), edges_reply(), *payload_replies())
    result = TdgBuilder(fake.client(), kv260).build(SPEC)
    assert result.warnings == []
    assert validate_tdg(result.tdg, kv260).is_valid
    assert result.tdg["nodes"] == FIXTURE["nodes"]
    assert result.tdg["edges"] == FIXTURE["edges"]
    assert result.tdg["constraints"] == {"deadline_s": 0.016667}
    assert result.tdg["graph"]["provenance"]["kind"] == "llm"


def test_spec_and_board_are_in_every_prompt(kv260):
    fake = FakeOllama(nodes_reply(), edges_reply(), *payload_replies())
    TdgBuilder(fake.client(), kv260).build(SPEC)
    for request in fake.requests:
        assert "Sobel edge detection on 1080p" in request["messages"][0]["content"]
        assert "Kria K26" in request["messages"][0]["content"]


def test_a_bad_formula_is_sent_back_to_its_step(kv260):
    bad = [dict(edge) for edge in FIXTURE["edges"]]
    fake = FakeOllama(
        nodes_reply(),
        reply({"edges": [{**bad[0], "bytes": "width*height"}, *bad[1:]]}),
        edges_reply(),
        *payload_replies(),
    )
    result = TdgBuilder(fake.client(), kv260).build(SPEC)
    assert result.tdg["edges"] == FIXTURE["edges"]
    assert "edge e1: bytes 'width*height'" in fake.requests[2]["messages"][-2]["content"]


def test_validator_errors_go_back_to_the_step_that_made_them(kv260):
    dangling = [dict(edge) for edge in FIXTURE["edges"]]
    dangling[3]["dst"] = "n9"
    bad_unroll = {"hints": {"unrollable": ["zz"]}}
    n3 = next(node for node in FIXTURE["nodes"] if node["id"] == "n3")
    fake = FakeOllama(
        nodes_reply(),
        reply({"edges": dangling}),
        payload_reply("n1"),
        payload_reply("n2"),
        payload_reply("n3", loop_nest={**n3["loop_nest"], **bad_unroll}),
        payload_reply("n4"),
        payload_reply("n5"),
        edges_reply(),  # fixes the R2 error
        payload_reply("n3"),  # fixes the R12 error
    )
    result = TdgBuilder(fake.client(), kv260).build(SPEC)
    assert result.tdg["nodes"] == FIXTURE["nodes"]
    fix_requests = [request["messages"][-2]["content"] for request in fake.requests[-2:]]
    assert "dst 'n9' does not exist" in fix_requests[0]
    assert "unknown loop variable 'zz'" in fix_requests[1]


def test_ops_mismatch_is_redone_once_then_left_as_a_warning(kv260):
    fake = FakeOllama(
        nodes_reply(),
        edges_reply(),
        payload_reply("n1"),
        payload_reply("n2", ops_claimed=1),
        *payload_replies()[2:],
        payload_reply("n2", ops_claimed=2),
    )
    result = TdgBuilder(fake.client(), kv260).build(SPEC)
    assert len(result.warnings) == 1 and result.warnings[0].startswith("n2:")
    assert "differ by more than" in fake.requests[-1]["messages"][-2]["content"]


def test_gives_up_when_errors_remain(kv260):
    dangling = [dict(edge) for edge in FIXTURE["edges"]]
    dangling[0]["src"] = "n9"
    fake = FakeOllama(
        nodes_reply(), *[reply({"edges": dangling})], *payload_replies(),
        *[reply({"edges": dangling})] * 3,
    )  # fmt: skip
    with pytest.raises(TdgBuildError, match="still invalid after 3 repairs"):
        TdgBuilder(fake.client(), kv260).build(SPEC)


def test_opaque_nodes_must_be_software_only(kv260):
    nodes = json.loads(nodes_reply()["content"])
    nodes["nodes"][0]["affinity"] = "any"
    fake = FakeOllama(reply(nodes), nodes_reply(), edges_reply(), *payload_replies())
    TdgBuilder(fake.client(), kv260).build(SPEC)
    assert "must be sw_only" in fake.requests[1]["messages"][-2]["content"]


def test_pipeline_agent_goes_from_task_to_graph(boards_dir, tmp_path):
    fake = FakeOllama(
        call("submit_spec", spec=SPEC), nodes_reply(), edges_reply(), *payload_replies()
    )
    agent = PipelineAgent(
        ["kv260"], boards_dir=boards_dir, client=fake.client(), run_dir=tmp_path, tools=[]
    )
    events = list(agent.handle(UserInput(text="Sobel on 1080p at 60 fps on the kv260")))
    assert [event.type for event in events] == [
        "assumptions", "spec_ready", "assistant_message", "tdg_ready",
    ]  # fmt: skip
    assert events[-1].tdg["nodes"] == FIXTURE["nodes"]
    assert read_json(tmp_path / "spec.json") == SPEC
    assert read_json(tmp_path / "tdg.json") == events[-1].tdg


def test_pipeline_agent_retries_the_graph_after_a_failure(boards_dir, tmp_path):
    fake = FakeOllama(call("submit_spec", spec=SPEC))  # then the model stops answering
    agent = PipelineAgent(
        ["kv260"], boards_dir=boards_dir, client=fake.client(), run_dir=tmp_path, tools=[]
    )
    first = list(agent.handle(UserInput(text="Sobel")))
    assert first[-1].type == "error" and "try again" in first[-1].message
    fake.replies = [nodes_reply(), edges_reply(), *payload_replies()]
    second = list(agent.handle(UserInput(text="again")))
    assert second[-1].type == "tdg_ready"


TINY_NET_NODES = [
    {"id": "n1", "name": "capture", "phase": "iteration", "affinity": "sw_only", "kind": "opaque"},
    {"id": "n2", "name": "tiny_net", "phase": "iteration", "affinity": "any", "kind": "template"},
    {"id": "n3", "name": "output", "phase": "iteration", "affinity": "sw_only", "kind": "opaque"},
]
TINY_NET_SPEC = {**SPEC, "networks": [{"name": "TinyNet", "gmacs": 0.1}]}


def conv_reply(c_in: int, c_out: int) -> dict:
    params = {"H": 64, "W": 64, "C_in": c_in, "C_out": c_out, "K": 3, "stride": 1, "dtype": "int8"}
    return reply({"template": {"op": "conv2d", "params": {**params, "padding": 1}}})


def tiny_net_replies(*conv_replies: dict) -> list[dict]:
    opaque = reply({"opaque": {"sw_time_s": 0.001, "source": "llm"}})
    edges = [
        {"id": "e1", "src": "n1", "dst": "n2", "kind": "data", "bytes": "64*64*3"},
        {"id": "e2", "src": "n2", "dst": "n3", "kind": "data", "bytes": "64*64*16"},
    ]
    nodes = [{**node, "purpose": "-"} for node in TINY_NET_NODES]
    first, *fixes = conv_replies
    return [
        reply({"graph_id": "tiny", "nodes": nodes}),
        reply({"edges": edges}),
        opaque,
        first,
        opaque,
        *fixes,
    ]


def test_network_nodes_are_rescaled_to_the_known_size(kv260):
    # 64*64*16*3*9 = 1.77M MACs, far below 0.1 GMACs; then 64*64*96*32*9 = 113M MACs
    fake = FakeOllama(*tiny_net_replies(conv_reply(3, 16), conv_reply(32, 96)))
    result = TdgBuilder(fake.client(), kv260).build(TINY_NET_SPEC)
    assert result.warnings == []
    assert result.tdg["nodes"][1]["template"]["params"]["C_out"] == 96
    nodes_request = fake.requests[0]["messages"][1]["content"]
    assert "TinyNet: 0.1 GMACs per inference" in nodes_request
    conv_request = fake.requests[3]["messages"][1]["content"]
    assert "TinyNet: 0.1 GMACs" in conv_request
    fix_request = fake.requests[-1]["messages"][-2]["content"]
    assert "add up to 0.00177 GMACs, but TinyNet has 0.1 GMACs" in fix_request
    assert "its share of the network is about 0.1 GMACs" in fix_request


def test_network_size_still_off_is_a_warning(kv260):
    fake = FakeOllama(*tiny_net_replies(conv_reply(3, 16), conv_reply(3, 32)))
    result = TdgBuilder(fake.client(), kv260).build(TINY_NET_SPEC)
    assert len(result.warnings) == 1 and result.warnings[0].startswith("network size:")
    assert not fake.replies  # rescaled once only


def test_no_network_size_check_without_networks_in_the_spec(kv260):
    fake = FakeOllama(*tiny_net_replies(conv_reply(3, 16)))
    result = TdgBuilder(fake.client(), kv260).build(SPEC)
    assert result.warnings == []
    assert "Neural network size" not in fake.requests[0]["messages"][1]["content"]


def test_an_implausibly_slow_node_is_redone(kv260):
    n3 = next(node for node in FIXTURE["nodes"] if node["id"] == "n3")
    loops = [{"var": "y", "range": 1078 * 10_000}, *n3["loop_nest"]["loops"][1:]]
    huge = payload_reply(
        "n3", loop_nest={**n3["loop_nest"], "loops": loops}, ops_claimed=n3["ops_claimed"] * 10_000
    )
    fake = FakeOllama(
        nodes_reply(),
        edges_reply(),
        *payload_replies()[:2],
        huge,
        *payload_replies()[3:],
        payload_reply("n3"),
    )
    result = TdgBuilder(fake.client(), kv260).build(SPEC)
    assert result.warnings == []
    assert result.tdg["nodes"] == FIXTURE["nodes"]
    assert "implausible" in fake.requests[-1]["messages"][-2]["content"]
