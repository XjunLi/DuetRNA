"""Recover gallery frames without changing the colors of the original renders."""
from pathlib import Path
import argparse, json, subprocess
from PIL import Image
import numpy as np
from prep_film import SAMPLES, SAMPLE_CROPS, SITE_VIDEOS, GALLERY_SIZE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ffmpeg', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent / 'build/gallery_original'
    counts = {}
    for nt, sample in SAMPLES:
        folder = root / nt
        folder.mkdir(parents=True, exist_ok=True)
        side, x, y = SAMPLE_CROPS[nt]
        pad = f'pad={side + 120}:{side + 120}:60:60:color=white,' if side == 1200 else ''
        vf = f'crop={side}:{side}:{x}:{y},{pad}fps=30,scale={GALLERY_SIZE}:{GALLERY_SIZE}:flags=lanczos'
        raw = subprocess.run([args.ffmpeg, '-hide_banner', '-loglevel', 'error', '-ss', '10',
            '-i', str(SITE_VIDEOS / f'{nt}_na_sample_{sample}.mp4'), '-t', '10.3', '-vf', vf,
            '-an', '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-'], capture_output=True, check=True).stdout
        frames = np.frombuffer(raw, np.uint8).reshape(-1, GALLERY_SIZE, GALLERY_SIZE, 3)
        for k, frame in enumerate(frames):
            Image.fromarray(frame).save(folder / f'f_{k:04d}.jpg', quality=96, subsampling=0)
        counts[nt] = len(frames)
        print(f'Original gallery {nt}: {len(frames)} frames', flush=True)
    (root / 'index.json').write_text(json.dumps(counts), encoding='utf-8')

if __name__ == '__main__':
    main()
