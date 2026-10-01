#!/usr/bin/env python3
"""Check the published markup and actual MP4 files, without media mocks."""
from __future__ import annotations
import base64
import functools
import hashlib
import http.server
import json
import re
import shutil
import threading
from pathlib import Path
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'qa/final-display';OUT.mkdir(parents=True,exist_ok=True)
class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self,*args):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Handler,directory=str(ROOT)))
threading.Thread(target=server.serve_forever,daemon=True).start()
base=f'http://127.0.0.1:{server.server_port}'
results=[]
def record(name):
    results.append(name);print('PASS:',name,flush=True)

def playback(page):
    try:
        page.wait_for_function('''()=>{const v=document.querySelector('#films video');return v&&v.readyState>=2&&v.currentTime>.2&&!v.paused}''',timeout=30000)
    except Exception:
        print(page.evaluate('''()=>({videos:[...document.querySelectorAll('video')].map(v=>({src:v.currentSrc,time:v.currentTime,paused:v.paused,error:v.error?.message})),status:[...document.querySelectorAll('.film-status')].map(e=>e.textContent)})'''),flush=True)
        page.screenshot(path=str(OUT/'failure.png'));raise

try:
    posters=json.loads((ROOT/'media/posters/manifest.json').read_text())['posters']
    assert len(posters)==30 and all(r['frame_index']==r['frame_count']-1 for r in posters)
    for row in posters:
        assert (ROOT/row['poster']).is_file()
        assert hashlib.sha256((ROOT/row['source']).read_bytes()).hexdigest()==row['source_sha256']
    record('All 30 posters use the final decoded frame; source MP4 hashes unchanged')
    structures=json.loads((ROOT/'media/structures/manifest.json').read_text())['structures']
    assert len(structures)==6
    for row in structures:
        source=(ROOT/row['source']).read_text()
        payload=re.search(r'atob\(\s*[\"\']([A-Za-z0-9+/=\s]+)[\"\']\s*\)',source)[1]
        assert hashlib.sha256(base64.b64decode(re.sub(r'\s','',payload))).hexdigest()==row['pdb_sha256']
        assert (ROOT/row['image']).is_file()
    record('Six structure thumbnails match the exact interactive-viewer PDB payloads')
    with sync_playwright() as p:
        chrome=shutil.which('google-chrome') or shutil.which('google-chrome-stable')
        browser=p.chromium.launch(headless=True,executable_path=chrome,args=['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'])
        page=browser.new_page(viewport={'width':1440,'height':1000})
        errors=[];requests=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('request',lambda r:requests.append(r.url))
        page.goto(base+'/',wait_until='networkidle')
        assert page.locator('.hero .hero-rna').count()==1
        assert page.locator('.hero [data-inline-video]').count()==0
        assert page.locator('video,iframe').count()==0
        assert page.locator('[data-inline-video]').count()==6
        assert page.locator('.hero-rna').evaluate('(i)=>i.complete&&i.naturalWidth>0')
        assert 'Georgia' in page.locator('body').evaluate('(e)=>getComputedStyle(e).fontFamily')
        assert page.locator('.footer-brand').count()==0
        order=page.locator('main > section').evaluate_all('(es)=>es.map(e=>e.id||e.className)')
        assert order.index('films')<order.index('explore')<order.index('overview'),order
        page.screenshot(path=str(OUT/'desktop.png'))
        record('Original static hero restored, serif typography retained; demonstrations precede research')
        for route in ('/','/demo/'):
            page.goto(base+route,wait_until='networkidle')
            for w in (320,360,390,680,768,950,1024,1200,1440):
                page.set_viewport_size({'width':w,'height':1000})
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),(route,w)
            record(f'Layout {route}: nine widths 320–1440 px without horizontal overflow')
        page.goto(base+'/',wait_until='networkidle')
        page.locator('#films').scroll_into_view_if_needed()
        page.wait_for_function('''()=>[...document.querySelectorAll('#films img')].every(i=>i.complete&&i.naturalWidth===640)''')
        page.locator('#films').screenshot(path=str(OUT/'films.png'))
        page.locator('#explore').scroll_into_view_if_needed()
        page.wait_for_function('''()=>[...document.querySelectorAll('.viewer-thumb img')].every(i=>i.complete&&i.naturalWidth===640)''')
        assert page.locator('.viewer-thumb img').count()==6
        page.locator('#explore').screenshot(path=str(OUT/'structures.png'))
        assert not any(urlparse(u).path.endswith('.mp4') for u in requests)
        assert page.locator('video,iframe').count()==0
        record('Final-frame video covers and actual 3D thumbnails load as images; no automatic video or WebGL startup')
        cards=page.locator('#films [data-film]')
        cards.nth(4).locator('[data-inline-video]').click();playback(page)
        assert page.locator('video').count()==1 and page.locator('dialog[open]').count()==0
        assert page.locator('video').evaluate('(v)=>v.currentTime<5')
        record('Real 120-nt video starts near time zero and plays inline, not from its poster time')
        cards.first.locator('[data-inline-video]').click();playback(page)
        assert page.locator('video').count()==1
        assert cards.nth(4).locator('.film-preview').is_visible()
        record('Changing samples releases the old video and restores its final-frame poster')
        page.locator('.site-footer').scroll_into_view_if_needed()
        page.wait_for_function('document.querySelector("video").paused')
        record('Playback pauses when the sample leaves the viewport')
        cards.first.locator('.film-reset').click()
        assert page.locator('video').count()==0
        record('Returning to preview removes the video decoder node')
        page.goto(base+'/demo/',wait_until='networkidle')
        assert page.locator('[data-inline-video]').count()==30
        for length in ('40','70','90','110','120','140'):
            page.locator(f'[data-filter-length="{length}"]').click()
            assert page.locator('[data-length]:visible').count()==5
        page.locator('[data-filter-length="all"]').click()
        assert page.locator('[data-length]:visible').count()==30
        page.screenshot(path=str(OUT/'gallery.png'))
        record('All 30 gallery final-frame previews and all six length filters remain available')
        page.set_viewport_size({'width':390,'height':1100});page.goto(base+'/',wait_until='networkidle')
        page.screenshot(path=str(OUT/'mobile.png'))
        page.locator('.menu').click();page.locator('#site-nav a[href="#films"]').click()
        assert page.locator('.menu').get_attribute('aria-expanded')=='false'
        page.locator('#films').screenshot(path=str(OUT/'mobile-films.png'))
        page.locator('#explore').scroll_into_view_if_needed()
        page.wait_for_function('''()=>[...document.querySelectorAll('.viewer-thumb img')].every(i=>i.complete&&i.naturalWidth>0)''')
        page.locator('#explore').screenshot(path=str(OUT/'mobile-structures.png'))
        record('Mobile menu and image-forward demonstration grids verified')
        page.set_viewport_size({'width':1440,'height':1000});page.goto(base+'/',wait_until='networkidle')
        page.route('**/*.mp4',lambda route:route.abort())
        page.locator('#films [data-inline-video]').first.click()
        page.wait_for_function('document.querySelector("#films .film-status").textContent.includes("Unable")')
        assert page.locator('video').count()==0
        assert page.locator('#films .film-preview').first.is_visible()
        record('Failed playback restores its finished-structure poster and retry control')
        assert not errors,errors
        record('No page-script exceptions')
        engine=browser.version;browser.close()
    (OUT/'report.json').write_text(json.dumps({'passed':results,'engine':engine,'browser':chrome or 'Chromium',
        'media':'Actual repository MP4s and real Mol* renders, no media mocks',
        'poster_bytes_total':sum(r['poster_bytes'] for r in posters),
        'structure_preview_bytes_total':sum(r['bytes'] for r in structures)},indent=2)+'\n')
finally:
    server.shutdown()
