"""Build TGFF 3.5 from the vendored source into build/tgff/ (outside third_party/, which stays unmodified).

Uses g++ with -O0 -fno-strict-aliasing: an -O2 build segfaults on the shipped series-parallel example; the -O0
build reproduces examples/kseries_parallel.tgff byte for byte (checked by `--check`).

Usage: python scripts/build_tgff.py [--cxx g++] [--check]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "third_party" / "tgff" / "tgff-3.5"
OUT = ROOT / "build" / "tgff"
UNITS = ["RGen", "Epsilon", "TGraph", "DBase", "TG", "ArgPack", "RMath", "psprint", "RStd", "main", "RString",
         "PGraph", "Interface", "Graph"]
FLAGS = ["-std=gnu++98", "-fpermissive", "-w", "-O0", "-fno-strict-aliasing"]


def exe_name() -> str:
    return "tgff.exe" if sys.platform == "win32" else "tgff"


def build(cxx: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for f in SRC.iterdir():
            if f.is_file():
                shutil.copy2(f, work / f.name)
        objs = []
        for u in UNITS:
            subprocess.run([cxx, *FLAGS, "-c", f"{u}.cc", "-o", f"{u}.o"], cwd=work, check=True)
            objs.append(f"{u}.o")
        subprocess.run([cxx, "-o", exe_name(), *objs], cwd=work, check=True)
        shutil.copy2(work / exe_name(), OUT / exe_name())
    return OUT / exe_name()


def check(exe: Path) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy2(SRC / "examples" / "kseries_parallel.tgffopt", tmp)
        subprocess.run([str(exe), "kseries_parallel"], cwd=tmp, check=True, capture_output=True)
        got = (Path(tmp) / "kseries_parallel.tgff").read_text().replace("\r", "")
    want = (SRC / "examples" / "kseries_parallel.tgff").read_text().replace("\r", "")
    return got == want


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cxx", default="g++")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    exe = build(args.cxx)
    print(f"built {exe.relative_to(ROOT)}")
    if args.check:
        ok = check(exe)
        print("reference example reproduced byte for byte" if ok else "MISMATCH with shipped example output")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
