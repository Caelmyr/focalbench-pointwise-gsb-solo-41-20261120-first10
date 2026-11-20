# FocalBench development verification scripts

These scripts are not part of the application. They exercise the anchor-based
registration workflow against temporary libraries and never touch real data.

Run every Python script in isolated mode with the project's package path:

```sh
python3 -I dev-tools/<script>.py
```

The scripts add `/opt/pointwise-env/site-packages` to `sys.path` themselves, so
no virtual environment is required.

## Backend checks

- `verify_anchor.py` — builds the engraved plaque demo end to end and asserts
  the three anchor states (full solve, translation only, unaligned).
- `http_smoke.py` — drives the Flask test client through settings,
  reconstruction and pair endpoints.
- `regression_demos.py` — confirms the two original demos reconstruct exactly
  as before when no anchor is configured.
- `check_csv_validation.py` — checks evidence CSV columns and anchor size
  validation.
- `check_portable.py` — confirms portable ZIP export/import keeps the anchor.
- `render_previews.py`, `render_frames.py`, `render_pairs.py` — produce visual
  proofs under this directory for manual inspection.

## Browser checks

`ui_smoke.py` and `ui_anchor_draw.py` use Playwright with the bundled Chromium
at `/opt/pointwise-env/browsers`. `stublibs/stubs.c` builds minimal avahi
stubs on first use (via `_browser.py`) so the browser can start in this
container; they are test-only and nothing in the application links against
them. A running server on port 18944 is required:

```sh
FOCALBENCH_DATA_DIR=dev-tools/runtime PORT=18944 \
  PYTHONPATH=/opt/pointwise-env/site-packages \
  python3 -c "from focalbench.web import main; main()"
```
