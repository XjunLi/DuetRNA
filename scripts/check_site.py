#!/usr/bin/env python3
"""Exercise the actual static site, poster files and MP4 decoders in Chrome."""
from __future__ import annotations
import functools
import http.server
import json
import shutil
import subprocess
import threading
from pathlib import Path
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'qa/editorial'
OUT.mkdir(parents=True, exist_ok=True)
class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass
server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(QuietHandler, directory=str(ROOT)))
threading.Thread(target=server.serve_forever, daemon=True).start()
base = f'http://127.0.0.1:{server.server_port}'
results = []
def record(name):
    results.append(name)
    print('PASS:', name, flush=True)

def playback(page, selector):
    try:
        page.wait_for_function('(selector) => {const v=document.querySelector(selector); return v && v.readyState >= 2 && v.currentTime > 0.2}', arg=selector, timeout=30000)
    except Exception:
        diagnostic = page.evaluate('''() => ({hidden: document.hidden, videos: [...document.querySelectorAll('video')].map(v => ({src:v.currentSrc, state:v.readyState, network:v.networkState, paused:v.paused, time:v.currentTime, error:v.error && {code:v.error.code, message:v.error.message}, rect:v.getBoundingClientRect().toJSON()})), statuses:[...document.querySelectorAll('.film-status')].map(e=>e.textContent)})''')
        print('PLAYBACK DIAGNOSTIC:', json.dumps(diagnostic), flush=True)
        page.screenshot(path=str(OUT / 'playback-failure.png'))
        raise

try:
    manifest = json.loads((ROOT / 'media/posters/manifest.json').read_text())['posters']
    assert len(manifest) == 30 and all(r['frame_index'] == 0 for r in manifest)
    assert all((ROOT / r['poster']).is_file() for r in manifest)
    record('30 genuine frame-zero poster files available')
    if shutil.which('ffprobe'):
        print(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,profile,pix_fmt,width,height','-of','json',str(ROOT/'demo/dual/120_na_sample_10.mp4')], text=True), flush=True)
    with sync_playwright() as p:
        chrome = shutil.which('google-chrome') or shutil.which('google-chrome-stable')
        browser = p.chromium.launch(headless=True, executable_path=chrome) if chrome else p.chromium.launch(headless=True, channel='chromium')
        print('BROWSER:', chrome or 'bundled Chromium', browser.version, flush=True)
        context = browser.new_context(viewport={'width': 1440, 'height': 1000})
        page = context.new_page(); errors = []; requests = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda request: requests.append(request.url))
        page.goto(base + '/', wait_until='networkidle')
        print('MP4 CODEC SUPPORT:', page.evaluate('document.createElement("video").canPlayType(\'video/mp4; codecs="avc1.42E01E"\')'), flush=True)
        assert page.locator('[data-inline-video]').count() == 7
        assert page.locator('video,iframe').count() == 0
        assert not any(urlparse(u).path.endswith('.mp4') for u in requests)
        assert page.locator('.hero [data-inline-video] img').evaluate('(i) => i.complete && i.naturalWidth === 640')
        assert 'Georgia' in page.locator('body').evaluate('(e) => getComputedStyle(e).fontFamily')
        assert not page.locator('.footer-brand').count()
        page.screenshot(path=str(OUT / 'desktop.png'))
        record('Home: real poster visible on arrival, serif body, no MP4 or viewer startup, no invented mark')
        for route in ['/', '/demo/']:
            page.goto(base + route, wait_until='networkidle')
            for width in [320, 360, 390, 680, 768, 950, 1024, 1200, 1440]:
                page.set_viewport_size({'width': width, 'height': 1000})
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), f'Overflow on {route} at {width}'
            record(f'Responsive layout: {route} at nine widths (320–1440 px)')
        page.goto(base + '/', wait_until='networkidle')
        page.locator('.hero [data-inline-video]').click()
        playback(page, '.hero video')
        assert page.locator('video').count() == 1 and page.locator('dialog[open]').count() == 0
        record('Real 120-nt MP4 decodes and plays inline after a click')
        page.locator('#films [data-inline-video]').first.click()
        playback(page, '#films video')
        assert page.locator('video').count() == 1
        assert page.locator('.hero .film-preview').is_visible()
        record('Switching sample releases the previous video and restores its poster')
        page.locator('#films .film-reset:visible').click()
        assert page.locator('video').count() == 0
        record('Back to preview removes the decoder node')
        page.set_viewport_size({'width': 390, 'height': 1050})
        page.goto(base + '/', wait_until='networkidle')
        page.screenshot(path=str(OUT / 'mobile.png'))
        page.locator('.menu').click()
        assert page.locator('.menu').get_attribute('aria-expanded') == 'true'
        page.locator('#site-nav a[href="#films"]').click()
        assert page.locator('.menu').get_attribute('aria-expanded') == 'false'
        record('Mobile menu opens and closes on navigation')
        requests.clear(); page.goto(base + '/demo/', wait_until='networkidle')
        assert page.locator('[data-inline-video]').count() == 30
        assert not any(urlparse(u).path.endswith('.mp4') for u in requests)
        for length in ['40', '70', '90', '110', '120', '140']:
            page.locator(f'[data-filter-length="{length}"]').click()
            assert page.locator('[data-length]:visible').count() == 5
        page.locator('[data-filter-length="all"]').click()
        assert page.locator('[data-length]:visible').count() == 30
        record('Gallery: all 30 previews and all six filters; no unsolicited MP4 loads')
        page.set_viewport_size({'width': 1440, 'height': 1000})
        page.goto(base + '/demo/', wait_until='networkidle')
        page.screenshot(path=str(OUT / 'gallery.png'))
        page.goto(base + '/', wait_until='networkidle')
        page.route('**/*.mp4', lambda route: route.abort())
        page.locator('.hero [data-inline-video]').click()
        page.wait_for_function('document.querySelector(".hero .film-status").textContent.includes("Unable")')
        assert page.locator('.hero .film-preview').is_visible() and page.locator('video').count() == 0
        record('Video-load failure restores the visible poster and retry path')
        assert errors == [], errors
        record('No page-script exceptions')
        engine = browser.version
        browser.close()
    (OUT / 'report.json').write_text(json.dumps({'passed': results, 'poster_bytes_total': sum(r['poster_bytes'] for r in manifest), 'engine': engine, 'browser': chrome or 'Chromium', 'media': 'Actual repository MP4 files, not media mocks'}, indent=2) + '\n')
finally:
    server.shutdown()
