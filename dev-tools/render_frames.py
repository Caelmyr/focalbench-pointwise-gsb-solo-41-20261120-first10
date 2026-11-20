"""Render end-frame proofs to visually confirm field-of-view masks and anchor coverage."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/opt/pointwise-env/site-packages")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from focalbench.demos import DEMOS, create_demo
from focalbench.imaging import read_capture, show_capture
from focalbench.store import Library


def main() -> None:
    with tempfile.TemporaryDirectory() as folder:
        library = Library(Path(folder))
        project = create_demo(library, "plaque", "preview plaque")
        picks = (0, 4, 7, 8)
        tiles = []
        for index in picks:
            capture = project["captures"][index]
            source = library.source_path(capture["sha"])
            decoded = read_capture(source, capture["name"])
            tile = show_capture(decoded, max_side=330)
            tiles.append((capture["name"], tile))
        width = sum(tile.width for _, tile in tiles) + 30
        height = max(tile.height for _, tile in tiles) + 24
        sheet = Image.new("RGBA", (width, height), (18, 21, 26, 255))
        x = 0
        for _, tile in tiles:
            sheet.alpha_composite(tile, (x, 18))
            x += tile.width + 10
        out = Path(__file__).resolve().parent / "plaque-frames.png"
        sheet.save(out)
        print("wrote", out, sheet.size)


if __name__ == "__main__":
    main()
