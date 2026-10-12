#!/usr/bin/env python3
"""Render the 1200 × 630 social preview from the site's own HTML and illustration."""
from pathlib import Path
import argparse
from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--browser', type=Path)
    args = parser.parse_args()
    options = {'headless': True}
    if args.browser:
        options['executable_path'] = str(args.browser.resolve())
    output = ROOT / 'media/duetrna-share.png'
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(**options)
        try:
            page = browser.new_page(viewport={'width': 1200, 'height': 630}, device_scale_factor=1)
            page.goto((ROOT / 'scripts/share-card.html').as_uri(), wait_until='load')
            page.evaluate('document.fonts.ready')
            page.wait_for_function('Array.from(document.images).every(im => im.complete && im.naturalWidth > 0)')
            page.screenshot(path=str(output), animations='disabled')
        finally:
            browser.close()
    with Image.open(output) as image:
        assert image.size == (1200, 630), image.size
    print(output)

if __name__ == '__main__':
    main()
