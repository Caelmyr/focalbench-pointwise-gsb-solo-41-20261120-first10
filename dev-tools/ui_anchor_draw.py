"""Exercise manually drawing an anchor rectangle and saving it through the UI."""

import sys
from pathlib import Path

sys.path.insert(0, "/opt/pointwise-env/site-packages")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from playwright.sync_api import sync_playwright

from _browser import ensure_browser_env


def main() -> None:
    ensure_browser_env()
    with sync_playwright() as toolkit:
        browser = toolkit.chromium.launch(
            executable_path="/opt/pointwise-env/browsers/chromium-1255/chrome-linux64/chrome",
            headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        page = browser.new_page(viewport={"width": 1500, "height": 1000})
        failures = []
        page.on("pageerror", lambda exc: failures.append(str(exc)))
        page.on("response", lambda resp: failures.append(f"HTTP {resp.status} {resp.url}")
                if resp.status >= 400 else None)

        page.goto("http://127.0.0.1:18944/", wait_until="networkidle")
        page.get_by_role("button", name="Load engraved serial plaque").click()
        page.wait_for_selector(".capture-row")

        page.get_by_role("button", name="Select patch").click()
        page.wait_for_selector("#stage-image")
        box = page.locator("#stage-image").bounding_box()
        page.mouse.move(box["x"] + box["width"] * 0.15, box["y"] + box["height"] * 0.25)
        page.mouse.down()
        page.mouse.move(box["x"] + box["width"] * 0.55, box["y"] + box["height"] * 0.55, steps=8)
        page.mouse.up()
        page.wait_for_selector(".readout-panel")

        page.locator("aside.right-pane").get_by_role("button", name="Reconstruction").click()
        page.wait_for_selector("#settings-form .anchor-box")
        page.get_by_role("button", name="Use selection as anchor").click()
        page.wait_for_selector(".notice")
        page.locator("#settings-form button[type=submit]").click()
        page.wait_for_selector(".notice:has-text('saved')")

        settings = page.evaluate("fetch('/api/stacks').then(r=>r.json())")
        stacks = {item["name"]: item for item in settings["stacks"]}
        # The most recently touched plaque stack.
        target = next(item for item in settings["stacks"] if "plaque" in item["name"].lower())
        detail = page.evaluate(f"fetch('/api/stacks/{target['id']}').then(r=>r.json())")
        anchor = detail["settings"]["anchor"]
        assert anchor and anchor["right"] - anchor["left"] > 0.2, anchor

        page.get_by_role("button", name="Clear anchor").click()
        page.wait_for_selector(".notice:has-text('cleared')")
        cleared = page.evaluate(f"fetch('/api/stacks/{target['id']}').then(r=>r.json())")
        assert cleared["settings"]["anchor"] is None

        assert not failures, failures
        browser.close()
        print("Manual anchor workflow passed.")


if __name__ == "__main__":
    main()
