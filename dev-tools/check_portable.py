"""Portable bundle round-trip must preserve the anchor recipe."""

import io
import sys
import tempfile
import zipfile
import json
from pathlib import Path

sys.path.insert(0, "/opt/pointwise-env/site-packages")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from focalbench.demos import create_demo
from focalbench.store import Library
from focalbench.transfer import portable_export, portable_import


def main() -> None:
    with tempfile.TemporaryDirectory() as folder:
        library = Library(Path(folder))
        project = create_demo(library, "plaque", "round trip plaque")
        stream, _ = portable_export(library, project["id"])
        stream.seek(0)
        raw = stream.read()
        with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
            manifest = json.loads(bundle.read("stack.json"))
        assert manifest["recipe"]["anchor"] == project["settings"]["anchor"]
        imported = portable_import(library, io.BytesIO(raw), "imported plaque")
        assert imported["settings"]["anchor"] == project["settings"]["anchor"]
        print("Portable anchor round trip passed:", imported["settings"]["anchor"])


if __name__ == "__main__":
    main()
