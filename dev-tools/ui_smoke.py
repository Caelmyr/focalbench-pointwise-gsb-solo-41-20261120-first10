"""Browser-level smoke test of the anchor workflow against a running local server."""

import sys
from pathlib import Path

sys.path.insert(0, "/opt/pointwise-env/site-packages")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from playwright.sync_api import sync_playwright

from _browser import CHROME, ensure_browser_env

BASE = "http://127.0.0.1:18944/"


def main() -> None:
    ensure_browser_env()
    with sync_playwright() as toolkit:
        browser = toolkit.chromium.launch(executable_path=CHROME, headless=True,
                                          args=["--no-sandbox", "--disable-dev-shm-usage"])
        page = browser.new_page(viewport={"width": 1500, "height": 1000})
        errors = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.on("response", lambda resp: errors.append(f"HTTP {resp.status} {resp.url}") if resp.status >= 400 else None)

        page.goto(BASE, wait_until="networkidle")
        page.get_by_role("button", name="Load engraved serial plaque").click()
        page.wait_for_selector(".capture-row", timeout=30000)
        page.locator("aside.right-pane").get_by_role("button", name="Reconstruction").click()
        page.wait_for_selector("#settings-form .anchor-box")
        anchor_summary = page.locator(".anchor-summary").first.inner_text()
        assert "Anchor set" in anchor_summary, anchor_summary
        page.screenshot(path=str(Path(__file__).parent / "ui-recipe.png"), full_page=True)

        page.get_by_role("button", name="Reconstruct", exact=True).first.click()
        page.wait_for_selector(".result-name", timeout=120000)
        page.get_by_role("button", name="Compare").click()
        page.wait_for_selector("#pair-form")
        page.get_by_role("button", name="Compare captures").click()
        page.wait_for_selector(".anchor-report", timeout=60000)
        report = page.locator(".anchor-report").inner_text()
        assert "translation only" in report and "not aligned" in report, report
        page.screenshot(path=str(Path(__file__).parent / "ui-compare.png"), full_page=True)

        # Planes table carries the anchor column.
        page.locator("aside.right-pane").get_by_role("button", name="Planes", exact=True).click()
        page.wait_for_selector("th")
        headers = [th.inner_text() for th in page.locator("th").all()]
        assert "Anchor" in headers, headers

        assert not errors, errors
        browser.close()
        print("Browser smoke test passed.")


if __name__ == "__main__":
    main()
