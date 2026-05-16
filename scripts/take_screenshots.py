"""Drive the Streamlit demo with Playwright and save screenshots.

Three screenshots:
  - docs/figures/01_report.png — Report Generation tab
  - docs/figures/02_qa.png     — RAG QA tab
  - docs/figures/03_about.png  — About tab

Doesn't call any generation API — captures the rendered layout, which is
what the assignment's "short report with screenshots" needs.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PORT = 8786
URL = f"http://127.0.0.1:{PORT}"


def _wait_for(port: int, timeout: float = 30.0) -> None:
    import socket

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"Streamlit didn't come up on :{port}")


def main() -> None:
    env = os.environ.copy()
    env["MEDXRAY_GENERATOR"] = "gemini"

    proc = subprocess.Popen(
        [
            str(ROOT / ".venv" / "bin" / "streamlit"),
            "run",
            str(ROOT / "app" / "streamlit_app.py"),
            "--server.headless",
            "true",
            "--server.port",
            str(PORT),
            "--browser.gatherUsageStats",
            "false",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        print(f"waiting for streamlit on :{PORT}...")
        _wait_for(PORT)
        time.sleep(3)  # let initial render settle

        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch()
            ctx = browser.new_context(viewport={"width": 1280, "height": 920})
            page = ctx.new_page()
            page.goto(URL)
            page.wait_for_load_state("networkidle", timeout=15000)
            time.sleep(2)

            # Report Generation tab (default)
            page.screenshot(path=FIG_DIR / "01_report.png", full_page=True)
            print(f"saved: {FIG_DIR / '01_report.png'}")

            # RAG QA tab
            for label in ("RAG QA", "RAG QA Mode", "QA"):
                try:
                    page.get_by_role("tab", name=label).click(timeout=2000)
                    break
                except Exception:
                    continue
            time.sleep(1.5)
            page.screenshot(path=FIG_DIR / "02_qa.png", full_page=True)
            print(f"saved: {FIG_DIR / '02_qa.png'}")

            # About tab
            for label in ("About / Models", "About"):
                try:
                    page.get_by_role("tab", name=label).click(timeout=2000)
                    break
                except Exception:
                    continue
            time.sleep(1.5)
            page.screenshot(path=FIG_DIR / "03_about.png", full_page=True)
            print(f"saved: {FIG_DIR / '03_about.png'}")

            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
