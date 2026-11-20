"""Save sample pair renders so the anchor overlay can be inspected visually."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/opt/pointwise-env/site-packages")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from focalbench.compare import pair_from_run, render_pair
from focalbench.reconstruction import Reconstruction
from focalbench.demos import create_demo
from focalbench.store import Library


def main() -> None:
    with tempfile.TemporaryDirectory() as folder:
        library = Library(Path(folder))
        project = create_demo(library, "plaque", "pair visual")
        run = Reconstruction(library).build(project["id"], "Visual run", None)
        full = library.run(project["id"], run["id"])
        ids = [row["id"] for row in full["summary"]["captures"]]
        pair = pair_from_run(library, full, ids[0], ids[-2], registered=True)
        out_dir = Path(__file__).resolve().parent
        for mode in ("split", "overlay", "difference", "side"):
            image = render_pair(pair, mode, 0.5)
            target = out_dir / f"plaque-pair-{mode}.png"
            image.convert("RGB").save(target)
            print("wrote", target.name, image.size)


if __name__ == "__main__":
    main()
