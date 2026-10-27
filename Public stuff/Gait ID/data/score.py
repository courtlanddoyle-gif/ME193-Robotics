"""Score both Gait ID models against the recorded logs in this folder.

    python data/score.py

Each .log is ~20 s of the board's live log while doing one activity (the
model1=/model2= lines printed by python/main.py with CALIBRATE_LOG on).
This re-runs classify_model1 / classify_model2 from python/main.py on the
logged features, so the numbers always match the current thresholds.
Runs on a laptop -- it only pulls the constants and classify functions out
of main.py, not the board-only imports.
"""
import ast
import re
from pathlib import Path

HERE = Path(__file__).parent
MAIN = HERE.parent / "python" / "main.py"

RECORDINGS = {  # file -> what was actually being done
    "stand.log": "standing",
    "walk.log": "walking",
    "jog.log": "running",
    "bikeback.log": "biking",   # lying on your back, air-biking
    "bikechair.log": "biking",  # sitting in a chair, pedaling
}

# The log uses shorter names for two features.
LOG_NAMES = {"gyro_mean": "gyro_mag_mean", "gyro_max": "gyro_mag_max"}


def load_classifiers():
    tree = ast.parse(MAIN.read_text(encoding="utf-8"))
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("classify_"):
            keep.append(node)
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            tuples = [t for t in node.targets if isinstance(t, ast.Tuple)]
            if tuples or any(n.startswith(("M1_", "M2_", "LABELS")) for n in names):
                keep.append(node)
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(MAIN), "exec"), ns)
    return ns


def load_rows(path):
    rows = []
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        if "model1=" not in line:
            continue
        rows.append({LOG_NAMES.get(k, k): float(v)
                     for k, v in re.findall(r"(\w+)=(-?\d+(?:\.\d+)?)\b", line)})
    return rows


def main():
    ns = load_classifiers()
    labels = ns["LABELS"]
    print(f"{'recording':<15} {'windows':>7} {'Model 1':>8} {'Model 2':>8}")
    for name, truth in RECORDINGS.items():
        rows = load_rows(HERE / name)
        m1 = sum(labels[ns["classify_model1"](r)] == truth for r in rows)
        m2 = sum(labels[ns["classify_model2"](r)] == truth for r in rows)
        print(f"{name:<15} {len(rows):>7} {100 * m1 / len(rows):>7.0f}% {100 * m2 / len(rows):>7.0f}%")


if __name__ == "__main__":
    main()
