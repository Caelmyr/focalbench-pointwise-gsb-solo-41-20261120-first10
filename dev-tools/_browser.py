"""Shared helpers for browser tests: ensure the avahi stub libraries exist."""

import os
import subprocess
from pathlib import Path

STUB_DIR = Path(__file__).resolve().parent / "stublibs"
LIBRARY_PATHS = (
    str(STUB_DIR),
    "/opt/pointwise-env/libs/usr/lib/x86_64-linux-gnu",
    "/opt/pointwise-env/libs/lib/x86_64-linux-gnu",
)
CHROME = "/opt/pointwise-env/browsers/chromium-1255/chrome-linux64/chrome"
SONAMES = ("libavahi-common.so.3", "libavahi-client.so.3")


def ensure_browser_env() -> None:
    for soname in SONAMES:
        target = STUB_DIR / soname
        if not target.exists():
            STUB_DIR.mkdir(exist_ok=True)
            subprocess.run(
                ["gcc", "-shared", "-fPIC", f"-Wl,-soname,{soname}",
                 "-o", str(target), str(STUB_DIR / "stubs.c")],
                check=True,
            )
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join([*LIBRARY_PATHS, existing]) if existing else ":".join(LIBRARY_PATHS)
