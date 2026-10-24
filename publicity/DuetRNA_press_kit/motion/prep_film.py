"""Prepare data for the DuetRNA motion film (film.html).

Inputs (all already in the press kit):
  media/sampling/sample_110.json   recorded 110-nt sampling trajectory, dual frames + local atom templates
  ../DuetRNA_site/videos/*.mp4     rotating renders of the six gallery samples

Outputs:
  motion/data/film_data.json       atoms, bonds, poses (sampling + training), hero nucleotide, canonical view
  motion/build/gallery/<nt>/f_XXXX.jpg   gallery frames, rainbow remapped to the film palette

Usage: python prep_film.py --ffmpeg <ffmpeg.exe> [--skip-gallery]
"""
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

HERE = Path(__file__).resolve().parent
KIT = HERE.parent
SITE_VIDEOS = KIT.parent / 'DuetRNA_site' / 'videos'

SAMPLES = [('40', '7'), ('70', '37'), ('90', '15'), ('110', '32'), ('120', '16'), ('140', '20')]
SAMPLE_CROPS = {'40': (1086, 56, 40), '70': (1200, 0, 0), '90': (1050, 74, 0),
                '110': (1130, 34, 28), '120': (1160, 20, 16), '140': (978, 110, 102)}
GALLERY_SIZE = 460

# 5'->3' gradient in the ball-and-stick palette (amber -> coral -> lavender -> blue -> teal).
GRADIENT = ['#D4854A', '#D06B60', '#8B72B0', '#5A8DB8', '#3A9E9C']


def hex_rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], dtype=float) / 255


def atom_class(name, group):
    if name == 'P':
        return 0
    if name.startswith('O'):
        return 1
    if name.startswith('N'):
        return 4
    return 2 if group == 0 else 3


def build_training_poses(final_poses, seed=20261015, num_states=101):
    """Construct exact conditional training interpolation states (linear R^3 + geodesic SO(3))."""
    n_res = final_poses.shape[0]
    rng = np.random.default_rng(seed)
    p1 = final_poses[:, :, :3]
    center = p1.reshape(-1, 3).mean(0, keepdims=True)
    p0 = rng.normal(0.0, 16.0, size=p1.shape)
    p0 -= p0.reshape(-1, 3).mean(0, keepdims=True)
    p0 += center
    q0 = Rotation.random(n_res * 2, random_state=rng).as_quat().reshape(n_res, 2, 4)
    q1 = final_poses[:, :, 3:]

    ts = np.linspace(0.0, 1.0, num_states)
    out = np.zeros((num_states, n_res, 2, 7), dtype=float)
    for i in range(n_res):
        for j in range(2):
            slerp = Slerp([0.0, 1.0], Rotation.from_quat([q0[i, j], q1[i, j]]))
            quats = slerp(ts).as_quat()
            for s, t in enumerate(ts):
                pos = (1.0 - t) * p0[i, j] + t * p1[i, j]
                out[s, i, j, :3] = pos
                out[s, i, j, 3:] = quats[s]
    return out


def prepare_structure(out):
    src = json.loads((KIT / 'media/sampling/sample_110.json').read_text())
    poses = np.asarray(src['poses'])  # (states, residues, 2[base,sugar], 7)
    residues = src['residues']
    atoms, world, res_atoms = [], [], []
    for i, r in enumerate(residues):
        idx = {}
        # group 0 = sugar (incl. phosphate), 1 = base; poses use j=0 base, j=1 sugar
        for g, kind, j in [(0, 'sugar', 1), (1, 'base', 0)]:
            pose = poses[-1, i, j]
            R = Rotation.from_quat(pose[3:]).as_matrix()
            for name, local in r[kind]['atoms'].items():
                idx[name] = len(atoms)
                atoms.append([i, g, *np.round(local, 4).tolist(), atom_class(name, g), name])
                world.append(R @ np.asarray(local) + pose[:3])
        res_atoms.append(idx)
    world = np.asarray(world)
    bonds = []
    for i, idx in enumerate(res_atoms):
        names = list(idx)
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                ia, ib = idx[names[a]], idx[names[b]]
                same_group = atoms[ia][1] == atoms[ib][1]
                glyco = {names[a], names[b]} == {"C1'", residues[i]['link']}
                if (same_group or glyco) and np.linalg.norm(world[ia] - world[ib]) < 1.9:
                    bonds.append([ia, ib])
        if i + 1 < len(res_atoms) and "O3'" in idx and 'P' in res_atoms[i + 1]:
            ia, ib = idx["O3'"], res_atoms[i + 1]['P']
            if np.linalg.norm(world[ia] - world[ib]) < 2.2:
                bonds.append([ia, ib])

    centre = world.mean(0)
    best = None
    for i, r in enumerate(residues):
        if r['type'] != 'A' or 'P' not in res_atoms[i]:
            continue
        pts = world[list(res_atoms[i].values())]
        c = pts.mean(0)
        radial = np.linalg.norm((c - centre)[[0, 2]])
        crowd = sum(np.linalg.norm(world - c, axis=1) < 7.0) - len(pts)
        score = radial - 0.04 * abs(c[1] - centre[1]) - 0.05 * crowd
        if best is None or score > best[0]:
            best = (score, i, c)
    _, hero, hero_c = best

    h = res_atoms[hero]
    P = world[h['P']]
    base_names = [n for n in residues[hero]['base']['atoms']]
    base_pts = world[[h[n] for n in base_names]]
    base_c = base_pts.mean(0)
    x = base_c - P
    x /= np.linalg.norm(x)
    _, _, vh = np.linalg.svd(base_pts - base_c)
    n = vh[-1] - np.dot(vh[-1], x) * x
    n /= np.linalg.norm(n)
    sugar_c = world[[h[k] for k in ("C1'", "C2'", "C3'", "C4'", "O4'")]].mean(0)
    up0 = np.cross(n, x)
    if np.dot(sugar_c - (P + base_c) / 2, up0) > 0:
        n = -n
        up0 = -up0
    tilt = np.deg2rad(38)
    z = np.cos(tilt) * n + np.sin(tilt) * up0
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    M = np.stack([x, y, z])
    q_canon = Rotation.from_matrix(M).as_quat().tolist()
    hero_centre = ((P + base_c) / 2 * 0.5 + sugar_c * 0.5).tolist()

    poses_training = build_training_poses(poses[-1])

    data = {
        'source': src['source'], 'source_sha256': src['source_sha256'], 'length': len(residues),
        'sequence': ''.join(r['type'] for r in residues), 'path_type': src['path_type'],
        'atoms': atoms, 'bonds': bonds,
        'residue_types': [r['type'] for r in residues], 'links': [r['link'] for r in residues],
        'poses': np.round(poses, 5).tolist(),
        'poses_training': np.round(poses_training, 5).tolist(),
        'hero': hero, 'hero_centre': hero_centre, 'q_canon': q_canon,
        'centre': centre.tolist(), 'extent': [world.min(0).tolist(), world.max(0).tolist()],
        'gradient': GRADIENT,
    }
    out.write_text(json.dumps(data, separators=(',', ':')), encoding='utf-8')
    print(f'atoms={len(atoms)} bonds={len(bonds)} hero={hero} ({residues[hero]["type"]}) '
          f'states={poses.shape[0]} -> {out}')


def recolor(rgb):
    """Map the renders' rainbow (5' red -> 3' cyan) onto the film's gradient, keeping shading."""
    f = rgb.astype(float) / 255
    mx, mn = f.max(-1), f.min(-1)
    delta = mx - mn
    sat = np.where(mx > 1e-6, delta / np.maximum(mx, 1e-6), 0)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    d = np.maximum(delta, 1e-6)
    hue = np.select([mx == r, mx == g], [((g - b) / d) % 6, (b - r) / d + 2], (r - g) / d + 4) * 60
    hue = np.where(hue > 300, 0, hue)
    u = np.clip(hue / 200, 0, 1)
    stops = np.stack([hex_rgb(c) for c in GRADIENT])
    pos = u * (len(stops) - 1)
    lo = np.floor(pos).astype(int).clip(0, len(stops) - 2)
    w = (pos - lo)[..., None]
    target = stops[lo] * (1 - w) + stops[lo + 1] * w
    sat_boost = np.where(sat > 0.08, np.clip(sat ** 0.78 * 1.15, 0, 1), sat)
    target = np.clip(target * 1.06, 0, 1)
    out = mx[..., None] * ((1 - sat_boost[..., None]) + sat_boost[..., None] * target)
    return np.clip(out * 255 + .5, 0, 255).astype(np.uint8)


def prepare_gallery(ffmpeg, root):
    from PIL import Image
    for nt, sample in SAMPLES:
        folder = root / nt
        folder.mkdir(parents=True, exist_ok=True)
        side, x, y = SAMPLE_CROPS[nt]
        pad = f'pad={side + 120}:{side + 120}:60:60:color=white,' if side == 1200 else ''
        vf = f'crop={side}:{side}:{x}:{y},{pad}fps=30,scale={GALLERY_SIZE}:{GALLERY_SIZE}:flags=lanczos'
        raw = subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-ss', '10', '-i',
                              str(SITE_VIDEOS / f'{nt}_na_sample_{sample}.mp4'), '-t', '10.3', '-vf', vf,
                              '-an', '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-'], capture_output=True, check=True).stdout
        frames = np.frombuffer(raw, np.uint8).reshape(-1, GALLERY_SIZE, GALLERY_SIZE, 3)
        for k, frame in enumerate(frames):
            Image.fromarray(recolor(frame)).save(folder / f'f_{k:04d}.jpg', quality=94)
        print(f'gallery {nt}: {len(frames)} frames')
    (root / 'index.json').write_text(json.dumps({nt: len(list((root / nt).glob('f_*.jpg'))) for nt, _ in SAMPLES}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ffmpeg', required=True)
    ap.add_argument('--skip-gallery', action='store_true')
    a = ap.parse_args()
    (HERE / 'data').mkdir(exist_ok=True)
    prepare_structure(HERE / 'data' / 'film_data.json')
    if not a.skip_gallery:
        prepare_gallery(a.ffmpeg, HERE / 'build' / 'gallery')
