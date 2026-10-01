#!/usr/bin/env python3
"""Create lightweight, genuine frame-zero posters without changing the source videos."""
from __future__ import annotations
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'media' / 'posters'

def run(args: list[str]) -> str:
    return subprocess.check_output(args, text=True).strip()

def main() -> None:
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        raise SystemExit('Install FFmpeg (including ffprobe) first.')
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    sources = sorted((ROOT / 'demo' / 'dual').glob('*.mp4')) + sorted((ROOT / 'videos').glob('*.mp4'))
    if len(sources) != 30:
        raise SystemExit(f'Expected 30 sample videos, found {len(sources)}; inspect the collection before rebuilding.')
    for source in sources:
        kind = 'dual' if source.parent.name == 'dual' else 'rainbow'
        destination = OUT / f'{kind}-{source.stem}.webp'
        # No seek: decode the actual first frame. Static posters never initialize a video decoder in the browser.
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(source),
                        '-frames:v', '1', '-vf', 'scale=640:-2:flags=lanczos',
                        '-c:v', 'libwebp', '-quality', '78', '-compression_level', '6',
                        '-threads', '1', str(destination)], check=True)
        probe = json.loads(run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                              '-show_entries', 'stream=width,height,duration', '-of', 'json', str(source)]))['streams'][0]
        row = {'source': source.relative_to(ROOT).as_posix(), 'poster': destination.relative_to(ROOT).as_posix(),
               'frame_index': 0, 'source_width': probe['width'], 'source_height': probe['height'],
               'duration_seconds': float(probe.get('duration', 0)), 'poster_bytes': destination.stat().st_size,
               'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
        rows.append(row)
        print(f"{row['poster']}: {row['poster_bytes']} bytes")
    (OUT / 'manifest.json').write_text(json.dumps({'description': 'First decoded video frame; frame_index=0. Resized to width 640, WebP quality 78.', 'posters': rows}, indent=2) + '\n')
    print(f"Generated {len(rows)} posters: {sum(row['poster_bytes'] for row in rows)} bytes total.")

if __name__ == '__main__':
    main()
