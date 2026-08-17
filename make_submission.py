"""Make the zip file of the submission and examine its contents.

The brief asks for one archive. The archive must contain the report, the two
CSV files of the predictions and the source code with its comments. The brief
also says that the archive must not contain the initial data sets.

This script makes that archive. The script does not write the archive if a
rule about the format is not obeyed. It examines these rules:

  * The file results_task1.csv exists. It has 1434 rows and no header. It
    contains three different labels: 0, 1 and the dummy label for spam.
  * The file results_task2.csv exists. It has 554 rows with 10 values in each
    row. Each value is in the range [0, 256]. Thus the values are at the
    original resolution and not at the resolution of 64x64 of the network.
  * The PDF file of the report exists.

The archive does not contain these items: the .npz data files, the .csv data
files, the directory .venv, the GloVe vectors, the model files, the caches,
and each other file that is not source code, a result or a figure.

To start the script, use the command:  python make_submission.py
"""
from __future__ import annotations

import os
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "submission.zip")

T1_CSV = os.path.join(HERE, "task1_nlp", "submission", "results_task1.csv")
T2_CSV = os.path.join(HERE, "task2_cv", "submission", "results_task2.csv")
REPORT = os.path.join(HERE, "REPORT.pdf")

SOURCE_DIRS = ["task1_nlp", "task2_cv", "worksheets"]
SOURCE_EXT = (".py", ".ipynb", ".png", ".csv", ".json", ".txt", ".md")
SKIP_DIRS = {"data", "__pycache__", ".ipynb_checkpoints", "models"}


def _fail(msg):
    print(f"[FAIL] {msg}")
    return False


def check_outputs() -> bool:
    ok = True
    if not os.path.exists(REPORT):
        ok = _fail("REPORT.pdf missing")

    if not os.path.exists(T1_CSV):
        ok = _fail("results_task1.csv missing -- run task1_nlp/run_task1.py")
    else:
        lines = open(T1_CSV).read().splitlines()
        vals = {float(x) for x in lines}
        if len(lines) != 1434:
            ok = _fail(f"results_task1.csv has {len(lines)} rows, expected 1434")
        if not lines[0].lstrip("-")[0].isdigit():
            ok = _fail("results_task1.csv appears to have a header row")
        if len(vals) != 3:
            ok = _fail(f"results_task1.csv has {len(vals)} distinct labels, "
                       f"expected 3 (0, 1 and the spam dummy): {sorted(vals)}")
        else:
            print(f"[ok] results_task1.csv: 1434 rows, labels {sorted(vals)}")

    if not os.path.exists(T2_CSV):
        ok = _fail("results_task2.csv missing -- run task2_cv/run_task2.py")
    else:
        pts = np.loadtxt(T2_CSV, delimiter=",")
        if pts.shape != (554, 10):
            ok = _fail(f"results_task2.csv is {pts.shape}, expected (554, 10)")
        elif not (0 <= pts.min() and pts.max() <= 256):
            ok = _fail(f"results_task2.csv values span [{pts.min():.1f}, "
                       f"{pts.max():.1f}] -- not original 256x256 resolution")
        else:
            print(f"[ok] results_task2.csv: 554 rows, range "
                  f"[{pts.min():.1f}, {pts.max():.1f}]")
    return ok


def collect():
    files = [REPORT]
    for root_name in SOURCE_DIRS:
        for root, dirs, names in os.walk(os.path.join(HERE, root_name)):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for n in names:
                if n.endswith(SOURCE_EXT):
                    files.append(os.path.join(root, n))
    files += [os.path.join(HERE, f) for f in
              ("README.md", "requirements.txt", "make_submission.py")
              if os.path.exists(os.path.join(HERE, f))]
    return files


def main() -> int:
    if not check_outputs():
        print("\nRefusing to build the archive: fix the failures above first.")
        return 1

    files = collect()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, os.path.relpath(f, HERE))
    size_mb = os.path.getsize(OUT) / 1e6
    print(f"\n[done] {OUT}: {len(files)} files, {size_mb:.1f} MB")
    leaked = [n for n in zipfile.ZipFile(OUT).namelist()
              if n.endswith(".npz") or "/data/" in n or n.endswith(".joblib")]
    if leaked:
        print(f"[FAIL] dataset or weight files leaked into the archive: {leaked}")
        return 1
    print("[ok] no datasets or model weights in the archive")
    return 0


if __name__ == "__main__":
    sys.exit(main())
