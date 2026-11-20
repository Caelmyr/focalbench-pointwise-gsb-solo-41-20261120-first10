"""End-to-end HTTP smoke test for anchor-based registration."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/opt/pointwise-env/site-packages")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from focalbench.web import create_app


def main() -> None:
    with tempfile.TemporaryDirectory() as folder:
        app = create_app(Path(folder))
        client = app.test_client()

        health = client.get("/api/health")
        assert health.status_code == 200, health.data
        assert health.get_json()["version"] == "1.1"

        demos = client.get("/api/demos").get_json()["demos"]
        assert any(item["id"] == "plaque" for item in demos), demos

        created = client.post("/api/stacks/demo", json={"demo": "plaque"}).get_json()
        stack_id = created["id"]
        assert created["settings"]["anchor"] is not None

        # Replace the anchor with a deliberately small region to exercise validation.
        patch = client.patch(
            f"/api/stacks/{stack_id}",
            json={"revision": created["revision"], "settings": {**created["settings"], "anchor": None}},
        )
        assert patch.status_code == 200, patch.data
        assert patch.get_json()["settings"]["anchor"] is None

        # Restore an anchor.
        anchor = {"left": 0.05, "top": 0.14, "right": 0.62, "bottom": 0.34}
        patch = client.patch(
            f"/api/stacks/{stack_id}",
            json={"revision": patch.get_json()["revision"],
                  "settings": {**created["settings"], "anchor": anchor}},
        )
        assert patch.status_code == 200, patch.data

        # Unknown settings key must still be rejected.
        bad = client.patch(
            f"/api/stacks/{stack_id}",
            json={"revision": patch.get_json()["revision"],
                  "settings": {**created["settings"], "bogus": 1}},
        )
        assert bad.status_code == 400, bad.data

        run = client.post(f"/api/stacks/{stack_id}/results",
                          json={"revision": patch.get_json()["revision"], "name": "Anchor run"})
        assert run.status_code == 201, run.data
        result = run.get_json()
        summary = result["summary"]
        assert summary["anchor"] == {"x": 34, "y": 64, "width": 394, "height": 94}, summary["anchor"]
        codes = {(w["code"]) for w in summary["warnings"]}
        assert "anchor-unaligned" in codes and "anchor-translation-only" in codes, codes
        states = [row["motion"].get("anchor_state") for row in summary["captures"]]
        assert states[-1] == "unaligned"

        # Pair endpoint with anchor overlay metadata and PNG render.
        first = summary["captures"][0]["id"]
        last = summary["captures"][-1]["id"]
        pair_json = client.post(
            f"/api/stacks/{stack_id}/results/{result['id']}/pair",
            json={"first": first, "second": last, "registered": True},
        )
        assert pair_json.status_code == 200, pair_json.data
        payload = pair_json.get_json()
        assert payload["anchor"]["states"] == ["translation-only", "unaligned"]

        for mode in ("split", "overlay", "difference", "side"):
            png = client.get(
                f"/api/stacks/{stack_id}/results/{result['id']}/pair.png"
                f"?first={first}&second={last}&alignment=registered&mode={mode}"
            )
            assert png.status_code == 200 and png.data[:4] == b"\x89PNG", mode

        # Raw pair has no anchor overlay.
        raw = client.post(
            f"/api/stacks/{stack_id}/results/{result['id']}/pair",
            json={"first": first, "second": last, "registered": False},
        )
        assert raw.get_json()["anchor"] is None

        # Forking preserves the anchor through portable settings.
        forked = client.post(f"/api/stacks/{stack_id}/fork", json={"name": "forked"})
        assert forked.status_code == 201, forked.data
        assert forked.get_json()["settings"]["anchor"] == anchor

        print("HTTP smoke test passed.")


if __name__ == "__main__":
    main()
