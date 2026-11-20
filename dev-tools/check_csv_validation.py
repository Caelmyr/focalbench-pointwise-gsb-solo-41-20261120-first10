"""Check evidence CSV columns and tiny-anchor validation behavior."""

import io
import csv
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, "/opt/pointwise-env/site-packages")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from focalbench.demos import create_demo
from focalbench.reconstruction import Reconstruction
from focalbench.store import Library
from focalbench.web import create_app


def main() -> None:
    with tempfile.TemporaryDirectory() as folder:
        app = create_app(Path(folder))
        client = app.test_client()
        created = client.post("/api/stacks/demo", json={"demo": "plaque"}).get_json()
        run = client.post(
            f"/api/stacks/{created['id']}/results",
            json={"revision": created["revision"], "name": "csv run"},
        ).get_json()
        evidence = client.get(f"/api/stacks/{created['id']}/results/{run['id']}/evidence")
        assert evidence.status_code == 200
        with zipfile.ZipFile(io.BytesIO(evidence.data)) as bundle:
            csv_name = next(name for name in bundle.namelist() if name.endswith("capture-evidence.csv"))
            rows = list(csv.DictReader(io.StringIO(bundle.read(csv_name).decode())))
        assert "anchor_state" in rows[0] and "anchor_coverage" in rows[0]
        states = {row["capture_filename"]: row["anchor_state"] for row in rows}
        assert states["plaque-09.png"] == "unaligned"
        assert states["plaque-01.png"] == "translation-only"
        print("CSV anchor columns:", [k for k in rows[0] if k.startswith("anchor")])

        # Regions under 0.5% are rejected by the region parser when the recipe is saved.
        tiny = client.patch(
            f"/api/stacks/{created['id']}",
            json={"revision": client.get(f"/api/stacks/{created['id']}").get_json()["revision"],
                  "settings": {**created["settings"],
                               "anchor": {"left": 0.5, "top": 0.5, "right": 0.501, "bottom": 0.501}}},
        )
        assert tiny.status_code == 400, tiny.data
        print("Sub-0.5% anchor rejected at save time:", tiny.get_json()["error"])

        # A region that passes parsing but is under 18 px at working size is rejected on build.
        narrow = client.patch(
            f"/api/stacks/{created['id']}",
            json={"revision": client.get(f"/api/stacks/{created['id']}").get_json()["revision"],
                  "settings": {**created["settings"],
                               "anchor": {"left": 0.5, "top": 0.5, "right": 0.512, "bottom": 0.512}}},
        )
        assert narrow.status_code == 200, narrow.data
        failed = client.post(
            f"/api/stacks/{created['id']}/results",
            json={"revision": narrow.get_json()["revision"], "name": "tiny anchor"},
        )
        assert failed.status_code == 422, failed.data
        assert "anchor" in failed.get_json()["error"].lower()
        print("Sub-18px anchor rejected at build time:", failed.get_json()["error"])
        print("CSV and validation checks passed.")


if __name__ == "__main__":
    main()
