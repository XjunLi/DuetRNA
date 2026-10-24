"""Build the silent 30.7-second DuetRNA motion film.

The animation is in motion/film.js. Install its Node dependencies with
`npm ci --prefix motion`. The existing reference inputs and prepared gallery
frames are reused. See motion/README.txt for data provenance and preparation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess

KIT = Path(__file__).resolve().parent
REVISION = 'motion-three-scenes-v27'
NAME = 'DuetRNA_31s.mp4'
CAPTIONS = [
    (0.0, 3.0, 'Introducing DuetRNA — co-designing RNA sequences and 3D structures.'),
    (4.6, 7.8, 'The sugar frame describes the sugar–phosphate backbone.'),
    (7.8, 10.8, 'DuetRNA adds a base frame for pairing and stacking.'),
    (14.5, 18.0, 'Both frames flow from noise to an RNA structure.'),
    (18.0, 25.0, 'Relative-pose supervision couples the two predicted endpoints.'),
]


def preview_html():
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DuetRNA · Motion film</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#f8f7f2;color:#24473d;font-family:Georgia,serif}}main{{max-width:1360px;margin:auto;padding:28px 30px}}header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:22px;gap:16px}}h1{{font-weight:400;font-size:30px;margin:0}}a{{color:#547b86;text-underline-offset:4px}}video{{width:100%;aspect-ratio:16/9;display:block;background:#f8f7f2}}nav{{display:flex;gap:10px;flex-wrap:wrap;margin:22px 0}}button{{border:1px solid #d6ded0;border-radius:5px;background:transparent;padding:10px 15px;font:16px Georgia,serif;color:#24473d;cursor:pointer}}button:hover{{background:#e8ecdf}}footer{{color:#788273;font-size:14px}}@media(max-width:600px){{main{{padding:18px 12px}}nav{{gap:7px}}button{{font-size:13px;padding:9px}}header a{{font-size:14px}}}}</style></head>
<body><main><header><h1>DuetRNA</h1><a href="{NAME}?v={REVISION}" download="{NAME}">Download MP4 ↓</a></header>
<video id="film" controls autoplay muted playsinline preload="auto" poster="media/demo_cover_en.jpg?v={REVISION}" src="{NAME}?v={REVISION}"></video>
<nav aria-label="Video chapters"><button data-time="0">01 · Dual-frame representation</button><button data-time="12.15">02 · Flow matching with constraint</button><button data-time="25">03 · Generated RNA samples</button></nav>
<footer>30.7 s · English · No audio · 1920 × 1080</footer></main><script>const film=document.querySelector('#film');document.querySelectorAll('[data-time]').forEach(b=>b.addEventListener('click',()=>{{film.currentTime=Number(b.dataset.time);film.play().catch(()=>{{}})}}));</script></body></html>'''


def stamp(s):
    ms=round(s*1000)
    return f'{ms//3600000:02d}:{ms//60000%60:02d}:{ms//1000%60:02d},{ms%1000:03d}'


def publish(output, build):
    (output/'media').mkdir(parents=True,exist_ok=True)
    poster=build/'frame_00.00.jpg'
    if poster.exists():
        shutil.copy2(poster,output/'media/demo_cover_en.jpg')
    (output/'preview.html').write_text(preview_html(),encoding='utf-8')
    (output/'DuetRNA_31s.srt').write_text('\n\n'.join(
        f'{i}\n{stamp(a)} --> {stamp(b)}\n{text}' for i,(a,b,text) in enumerate(CAPTIONS,1))+'\n',encoding='utf-8')
    trajectory=json.loads((KIT/'motion/data/film_data.json').read_text(encoding='utf-8'))
    manifest={
        'revision':REVISION,'video':NAME,'duration_seconds':922/30,'fps':30,
        'resolution':[1920,1080],'language':'en','audio':False,
        'modules':[
            {'start':0,'end':12.15,'name':'Dual-frame representation'},
            {'start':12.15,'end':25,'name':'Flow matching with relative-pose supervision'},
            {'start':25,'end':922/30,'name':'Generated RNA samples fading out'}],
        'source':trajectory['source'],'source_sha256':trajectory['source_sha256'],
        'path_type':trajectory['path_type'],'saved_states':len(trajectory['poses']),
        'interpolation':'Only between adjacent recorded poses for smooth display.',
        'camera':'Display framing adapts to each recorded flow state; its RNA coordinates are not altered.',
        'representation_transition':'The pullback (10.8-12.15 s) overlaps leftward travel (11.25-12.65 s). The RNA holds, becomes paired frames (12.95-13.22 s), then holds through 13.95 s (0.73 s). Each frame follows a slightly staggered curved display-space path into its exact first recorded noise pose at 14.5 s, immediately handing off to forward flow. The camera and baseline arrow size remain fixed during this initialization. Forward sampling begins at 14.5 s with a smooth startup that rejoins the original clock within 0.1032 s. There is no stationary noise hold.',
        'frame_visibility':'Six flat arrows projected from the actual 3D axes overlay the atoms in the close-up. Overview frame meshes also render above atoms. Sugar label is on the left, base label on the right; each has one straight leader.',
        'snapshot_timing':'Checkpoints remain immediately on arrival (14.5, 17.3667, 20.2333 s); no delayed reveal.',
        'scatter_effect':'Individual deterministic stagger, continuous acceleration and deceleration compressed into 0.55 s, with a restrained curved path and a maximum 8% per-frame size variation. No whole-group contraction, scale overshoot, rebound, radial streaks or fake trails. This is display-only initialization; the original noise poses and subsequent saved sampling path are unchanged.',
        'motion_blur':{'interval':[13.95,14.5],'temporal_samples':8,'shutter_seconds':1/60,'scope':'Moving 3D frames only; labels, subtitles and the time axis stay sharp.'},
        'ending':'Six samples rotate once, then begin fading at 30 s. The last eight near-white frames are trimmed; the movie ends at 922/30 seconds, with the last retained frame at 30.7 s. No closing title card.',
        'subtitle_font_px':42,
        'transition_labels':'Flow labels fade out over 24.7-25.0 s before the gallery heading appears; wording and total timing unchanged.',
        'representation':'Actual nucleotide 36 in the 110-nt sample; sugar C4 anchor and base N9 anchor.',
        'constraint_inset':'The relative-pose diagram is extracted directly from the paper method figure, without formulas or an enclosing box. Original red and blue axes and rotation/translation arrows are retained. It illustrates endpoint-frame supervision, not measured errors or a hard locking step during inference. A fixed dashed brace groups the trajectory.',
        'colors':{'sugar_frame':'#d72c40','base_frame':'#204fe0'},
        'gallery':{'sample_lengths':[40,70,90,110,120,140],'source_frames':[0,308],
                   'rotation_seconds':[25.4,30.0],'palette':'Original RGB frames recovered from the six source videos, without hue remapping.'},
        'captions':CAPTIONS,
    }
    (KIT/'motion/timeline.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Preview:',output/'preview.html')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=KIT)
    p.add_argument('--build-dir',type=Path,default=KIT/'motion/build/v27')
    p.add_argument('--ffmpeg')
    p.add_argument('--node',default=shutil.which('node') or 'node')
    p.add_argument('--preview-only',action='store_true',help='Render and inspect key frames without encoding.')
    p.add_argument('--publish-only',action='store_true',help='Refresh preview, subtitles and provenance for an existing export.')
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    if not a.publish_only:
        cmd=[a.node,str(KIT/'motion/render_film.cjs'),'--build',str(a.build_dir),'--output',str(a.output_dir/NAME)]
        if a.ffmpeg:cmd+=['--ffmpeg',a.ffmpeg]
        if not a.preview_only:cmd+=['--full']
        subprocess.run(cmd,check=True,cwd=KIT/'motion')
    publish(a.output_dir,a.build_dir)


if __name__=='__main__':
    main()
