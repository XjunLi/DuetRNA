#!/usr/bin/env python3
"""Extract each video's actual final decoded frame; never modify an MP4."""
from __future__ import annotations
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'media/posters'

def main() -> None:
    if not all(shutil.which(tool) for tool in ('ffmpeg', 'ffprobe')):
        raise SystemExit('Install FFmpeg and ffprobe first.')
    sources = sorted((ROOT / 'demo/dual').glob('*.mp4')) + sorted((ROOT / 'videos').glob('*.mp4'))
    if len(sources) != 30:
        raise SystemExit(f'Expected 30 sample videos, found {len(sources)}.')
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in sources:
        probe = json.loads(subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
            '-show_entries', 'stream=width,height,duration,nb_read_frames',
            '-of', 'json', str(source)], text=True))['streams'][0]
        count = int(probe['nb_read_frames'])
        if count < 1:
            raise ValueError(f'No decoded frames: {source}')
        kind = 'dual' if source.parent.name == 'dual' else 'rainbow'
        dest = OUT / f'final-{kind}-{source.stem}.webp'
        # Selecting by decoded frame index avoids duration rounding and seek errors.
        subprocess.run([
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-threads', '2',
            '-i', str(source), '-vf', f'select=eq(n\\,{count-1}),scale=640:-2:flags=lanczos',
            '-frames:v', '1', '-c:v', 'libwebp', '-quality', '82',
            '-compression_level', '6', '-threads', '1', str(dest)], check=True)
        if not dest.is_file() or dest.stat().st_size < 100:
            raise ValueError(f'Missing or empty poster: {dest}')
        row = dict(source=source.relative_to(ROOT).as_posix(),
                   poster=dest.relative_to(ROOT).as_posix(), frame_index=count-1,
                   frame_count=count, source_width=probe['width'], source_height=probe['height'],
                   duration_seconds=float(probe.get('duration', 0)), poster_bytes=dest.stat().st_size,
                   source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        rows.append(row)
        print(f"{row['poster']}: frame {count-1}/{count}, {row['poster_bytes']} bytes", flush=True)
    (OUT / 'manifest.json').write_text(json.dumps({
        'description': 'Actual final decoded video frame. Playback remains unchanged and begins at time zero.',
        'posters': rows}, indent=2) + '\n')

if __name__ == '__main__':
    main()
