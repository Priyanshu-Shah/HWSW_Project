"""Locations of the shared folders (we only ever read from them)."""

from pathlib import Path

# .../pre_TDG/src/hwsw/repo_paths.py -> repository root
REPO_ROOT = Path(__file__).resolve().parents[3]

TDG_SCHEMA_PATH = REPO_ROOT / "TDG" / "schemas" / "tdg.schema.json"
TDG_EXAMPLES_DIR = REPO_ROOT / "TDG" / "schemas" / "examples"
LIBRARY_BOARDS_DIR = REPO_ROOT / "library" / "boards"

# Test fixtures that the stub conversation agent and the app fall back to until the library
# team delivers real board files. Remove these (and the fallback) once library/boards exists.
FIXTURES_DIR = REPO_ROOT / "pre_TDG" / "tests" / "fixtures"
FIXTURE_TDGS_DIR = FIXTURES_DIR / "tdgs"
FIXTURE_BOARDS_DIR = FIXTURES_DIR / "platforms"
