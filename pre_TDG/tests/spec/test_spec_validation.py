import copy

import pytest

from hwsw.spec import validate_spec

VALID_SPEC = {
    "task_summary": "Real-time object detection on 720p video",
    "platform": "zcu104",
    "arrival": {"kind": "streaming", "period_s": 0.0333},
    "timing": {"deadline_s": 0.0333, "hardness": "hard", "derived_from": "30 fps, sequential (D3)"},
    "assumptions": [
        {
            "key": "model",
            "value": "YOLOv8n, 8.7 GFLOPs @640",
            "source": "web",
            "url": "https://example.org/yolov8",
            "confidence": "medium",
        },
        {"key": "dtype", "value": "int8", "source": "default", "confidence": "low"},
    ],
    "open_questions": [],
}


@pytest.fixture
def spec() -> dict:
    return copy.deepcopy(VALID_SPEC)


def test_valid_spec(spec):
    assert validate_spec(spec, ready_for_tdg_builder=True).is_valid


def test_web_assumption_needs_a_source_url(spec):
    spec["assumptions"][0]["url"] = ""
    assert not validate_spec(spec).is_valid


def test_hard_timing_needs_a_deadline(spec):
    del spec["timing"]["deadline_s"]
    assert not validate_spec(spec).is_valid


def test_no_hardness_means_no_deadline(spec):
    spec["timing"] = {"hardness": "none", "derived_from": "no timing requirement stated"}
    assert validate_spec(spec).is_valid
    spec["timing"]["deadline_s"] = 0.01
    assert not validate_spec(spec).is_valid


def test_unanswered_questions_only_block_the_tdg_builder(spec):
    spec["open_questions"] = ["What frame rate do you need?"]
    assert validate_spec(spec).is_valid
    report = validate_spec(spec, ready_for_tdg_builder=True)
    assert not report.is_valid
    assert "What frame rate do you need?" in report.format()
