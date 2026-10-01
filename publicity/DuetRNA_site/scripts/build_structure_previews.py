#!/usr/bin/env python3
"""Render six thumbnails from the exact PDB payloads used by the interactive viewers."""
from __future__ import annotations
import base64
import functools
import hashlib
import http.server
import json
import re
import shutil
import threading
import urllib.request
from pathlib import Path
from PIL import Image, ImageChops
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'qa/structure-previews'
OUT = ROOT / 'media/structures'
SAMPLES = {'40':'40nt_genonly.html','70':'70nt_genonly.html','90':'90nt_genonly.html',
           '110':'110nt_genonly.html','120':'120nt_s16_genonly.html','140':'140nt_genonly.html'}

PAGE = '''<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="molstar.css"><style>
html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#fffefa}
#viewport{position:absolute;inset:0}.msp-viewport-controls,.msp-viewport-top-left-controls,
.msp-highlight-toast-wrapper,.msp-toast-container{display:none!important}
</style></head><body><div id="viewport"></div><script src="molstar.js"></script><script>
(async()=>{try{
 const id=new URLSearchParams(location.search).get('id');
 const pdb=await (await fetch(id+'.pdb')).text();
 const viewer=await molstar.Viewer.create('viewport',{
 extensions:[],layoutIsExpanded:false,layoutShowControls:false,layoutShowRemoteState:false,
 layoutShowSequence:false,layoutShowLog:false,layoutShowLeftPanel:false,
 viewportShowExpand:false,viewportShowControls:false,viewportShowSettings:false,
 viewportShowSelectionMode:false,viewportShowAnimation:false,volumeStreamingDisabled:true,pixelScale:1});
 viewer.plugin.canvas3d.setProps({renderer:{backgroundColor:0xfffefa}});
 await viewer.loadStructureFromData(pdb,'pdb');
 // Use the same molecular representation as the interactive viewer; no substitute sample.
 viewer.plugin.managers.camera.reset();
 await new Promise(resolve=>setTimeout(resolve,1200));
 document.documentElement.dataset.ready='true';
 window.previewViewer=viewer;
}catch(e){document.documentElement.dataset.error=e.message;console.error(e);}})();
</script></body></html>'''

class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ('js','css'):
        error = None
        for host in ('https://cdn.jsdelivr.net/npm/molstar@3.2.0/build/viewer/',
                     'https://unpkg.com/molstar@3.2.0/build/viewer/'):
            try:
                with urllib.request.urlopen(host+'molstar.'+suffix, timeout=60) as response:
                    (WORK/('molstar.'+suffix)).write_bytes(response.read())
                break
            except Exception as exc:
                error = exc
        else:
            raise RuntimeError(f'Unable to download pinned Mol* {suffix}: {error}')
    (WORK/'render.html').write_text(PAGE)
    rows=[]
    for length, filename in SAMPLES.items():
        source=ROOT/'viewer'/filename
        match=re.search(r'atob\(\s*[\"\']([A-Za-z0-9+/=\s]+)[\"\']\s*\)', source.read_text())
        if not match: raise ValueError(f'No PDB payload: {source}')
        pdb=base64.b64decode(re.sub(r'\s','',match[1]))
        assert b'ATOM ' in pdb
        (WORK/f'{length}.pdb').write_bytes(pdb)
        rows.append(dict(length=int(length), source=source.relative_to(ROOT).as_posix(),
                         pdb_sha256=hashlib.sha256(pdb).hexdigest(),
                         image=f'media/structures/{length}nt.webp'))
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Handler,directory=str(WORK)))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    try:
        with sync_playwright() as p:
            chrome=shutil.which('google-chrome') or shutil.which('google-chrome-stable')
            browser=p.chromium.launch(headless=True,executable_path=chrome,
                args=['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'])
            for row in rows:
                page=browser.new_page(viewport={'width':640,'height':480},device_scale_factor=1)
                errors=[];page.on('pageerror',lambda e: errors.append(str(e)))
                page.goto(f'http://127.0.0.1:{server.server_port}/render.html?id={row["length"]}',wait_until='load')
                page.wait_for_function("document.documentElement.dataset.ready || document.documentElement.dataset.error",timeout=90000)
                error=page.evaluate('document.documentElement.dataset.error')
                if error: raise RuntimeError(error)
                page.mouse.move(0,0)
                page.wait_for_timeout(250)
                png=WORK/f'{row["length"]}nt.png'
                page.screenshot(path=str(png))
                image=Image.open(png).convert('RGB')
                diff=ImageChops.difference(image,Image.new('RGB',image.size,(255,254,250)))
                if not diff.getbbox(): raise RuntimeError('Blank molecular preview')
                image.save(ROOT/row['image'],'WEBP',quality=86,method=6)
                row.update(renderer='Mol* 3.2.0, original viewer PDB, default representation',
                           width=640,height=480,bytes=(ROOT/row['image']).stat().st_size)
                print(f"Rendered {row['length']} nt from {row['source']}: {row['bytes']} bytes",flush=True)
                assert not errors,errors
                page.close()
            browser.close()
    finally:
        server.shutdown()
    (OUT/'manifest.json').write_text(json.dumps({'description':'Static renders of the matching interactive structures. No video sample substitution.','structures':rows},indent=2)+'\n')

if __name__=='__main__':
    main()
