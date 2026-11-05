"""Capture a screenshot of a running Streamlit app for the book's figures.
Usage: python3 screenshot.py URL OUT.png [click-text] [width height]"""
import sys, time
from playwright.sync_api import sync_playwright

url, out = sys.argv[1], sys.argv[2]
click = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None
w, h = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (1300, 900)
with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": w, "height": h}, device_scale_factor=2)
    page.goto(url)
    page.wait_for_selector("h1, h2", timeout=120000)
    time.sleep(6)
    for action in (click.split("|") if click else []):
        if action.startswith("type:"):                     # type into the chat box and send
            box = page.get_by_role("textbox").first
            box.fill(action[5:]); box.press("Enter"); time.sleep(5)
        else:
            page.get_by_text(action, exact=True).first.click(); time.sleep(4)
    page.screenshot(path=out)
    b.close()
