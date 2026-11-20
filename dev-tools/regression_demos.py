"""Regression check: the two original demos still reconstruct without anchors."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/opt/pointwise-env/site-packages")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from focalbench.demos import create_demo
from focalbench.reconstruction import calculate
from focalbench.store import Library


def main() -> None:
    with tempfile.TemporaryDirectory() as folder:
        library = Library(Path(folder))
        for demo_id in ("circuit", "shim"):
            project = create_demo(library, demo_id, f"regression {demo_id}")
            snapshot = library.snapshot(project["id"])
            result = calculate(library, snapshot)
            summary = result.summary
            assert summary["anchor"] is None
            for row in summary["captures"]:
                motion = row["motion"]
                assert motion["anchor_state"] is None
                assert motion["anchor_coverage"] is None
            print(demo_id, "ok",
                  "frames", summary["capture_count"],
                  "reliable", round(summary["reliable_fraction"], 3),
                  "warnings", [w["code"] for w in summary["warnings"]])
        print("Regression passed.")


if __name__ == "__main__":
    main()
