"""Runs the pure JavaScript modules' tests (tests/js) with Node's built-in test runner."""

import shutil
import subprocess
from pathlib import Path

import pytest

JS_TESTS_DIR = Path(__file__).parent / "js"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_javascript_unit_tests_pass():
    result = subprocess.run(
        ["node", "--test", *map(str, sorted(JS_TESTS_DIR.glob("*.test.mjs")))],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
