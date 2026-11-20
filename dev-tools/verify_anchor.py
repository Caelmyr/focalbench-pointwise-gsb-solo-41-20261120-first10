"""Exercise anchor-based alignment on the synthetic plaque target.

Run with the isolated interpreter: python3 -I dev-tools/verify_anchor.py
"""

import sys
import tempfile
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, "/opt/pointwise-env/site-packages")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from focalbench.demos import create_demo
from focalbench.models import RunSettings
from focalbench.reconstruction import calculate
from focalbench.store import Library


def main() -> None:
    with tempfile.TemporaryDirectory() as folder:
        library = Library(Path(folder))
        project = create_demo(library, "plaque", "Anchor verification plaque")
        snapshot = library.snapshot(project["id"])

        anchored = calculate(library, snapshot)
        print("=== With anchor ===")
        for row in anchored.summary["captures"]:
            motion = row["motion"]
            print(
                f"{row['name']} pos={row['position']:6.1f} "
                f"dx={motion['dx']:7.2f} dy={motion['dy']:7.2f} scale={motion['scale']:.4f} "
                f"state={str(motion.get('anchor_state')):18s} coverage={motion.get('anchor_coverage')} "
                f"score={motion['score']}"
            )
        print("\nWarnings:")
        for item in anchored.summary["warnings"]:
            print(" -", item["code"], item["detail"])
        print("\nAnchor box (processed px):", anchored.summary["anchor"])
        print("Crop:", anchored.summary["crop"])

        states = [row["motion"].get("anchor_state") for row in anchored.summary["captures"]]
        reference_index = next(
            index for index, row in enumerate(anchored.summary["captures"]) if row["is_reference"]
        )
        # Frame 1 and frame 8 retain only part of the engraving; frame 9 loses it entirely.
        assert states[0] == "translation-only", states
        assert states[-1] == "unaligned", states
        assert states[-2] == "translation-only", states
        assert states[1] == "full", states
        assert all(state == "full" for index, state in enumerate(states)
                   if index not in (0, 1, 7, 8, reference_index)), states
        assert states[reference_index] is None
        scales = [
            row["motion"]["scale"]
            for index, row in enumerate(anchored.summary["captures"])
            if index not in (0, 7, 8, reference_index)
        ]
        assert all(abs(scale - 1.0) > 1e-9 for scale in scales), "breathing scales should be solved"
        for partial in (0, 7):
            assert anchored.summary["captures"][partial]["motion"]["scale"] == 1.0
        unaligned = anchored.summary["captures"][-1]["motion"]
        assert unaligned["dx"] == 0.0 and unaligned["dy"] == 0.0
        assert unaligned["anchor_coverage"] == 0.0

        no_anchor_settings = replace(RunSettings.parse(snapshot["settings"]), anchor=None)
        snapshot_no_anchor = {**snapshot, "settings": no_anchor_settings.public()}
        full_frame = calculate(library, snapshot_no_anchor)
        print("\n=== Full-frame alignment (no anchor) ===")
        for row in full_frame.summary["captures"]:
            motion = row["motion"]
            print(
                f"{row['name']} dx={motion['dx']:7.2f} dy={motion['dy']:7.2f} "
                f"scale={motion['scale']:.4f} score={motion['score']:.3f} warning={motion['warning']}"
            )

        anchored_middle = anchored.summary["captures"][4]["motion"]
        loose_middle = full_frame.summary["captures"][4]["motion"]
        print("\nMiddle frame anchor solve:", anchored_middle["dx"], anchored_middle["dy"], anchored_middle["scale"])
        print("Middle frame full-frame solve:", loose_middle["dx"], loose_middle["dy"], loose_middle["scale"])

        # Pair comparison carries the anchor metadata.
        from focalbench.compare import compare_metrics, pair_from_run, render_pair
        from focalbench.reconstruction import Reconstruction

        published = Reconstruction(library).build(project["id"], "Verification run", None)
        run_id = published["id"]
        run = library.run(project["id"], run_id)
        first_id = run["summary"]["captures"][0]["id"]
        last_id = run["summary"]["captures"][-1]["id"]
        pair = pair_from_run(library, run, first_id, last_id, registered=True)
        metrics = compare_metrics(pair, None)
        print("\nPair anchor payload:", metrics["anchor"])
        assert metrics["anchor"] is not None
        image = render_pair(pair, "split", 0.5)
        print("Pair render size:", image.size, image.mode)
        print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
