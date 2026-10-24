"""Extract only the relative-pose diagram from the paper's method figure."""
from pathlib import Path
from PIL import Image
import numpy as np
import hashlib, json

MOTION = Path(__file__).resolve().parent
SOURCE = MOTION.parent / 'media/dual_frame_flow_matching.png'
OUTPUT = MOTION / 'assets/relative_pose_paper.png'
# Original pixel coordinates. The lower-right notch excludes the translation
# formula while retaining the entire lower-left red arrowhead.
box = (2080, 604, 2675, 982)
rgb = np.asarray(Image.open(SOURCE).convert('RGB').crop(box)).astype(float)
keep = np.ones(rgb.shape[:2], dtype=bool)
keep[928-box[1]:, 2160-box[0]:] = False
# Remove the figure's uniform 251/251/251 paper background, preserving the
# original anti-aliased arrow pixels rather than repainting their geometry.
alpha = np.clip((251-rgb.min(axis=2))/251, 0, 1)
alpha[(rgb.min(axis=2) >= 249) | ~keep] = 0
color = np.zeros_like(rgb)
np.divide(rgb-251, alpha[..., None], out=color, where=alpha[..., None] > 0)
color = np.clip(color+251, 0, 255)
rgba = np.dstack((np.rint(color), np.rint(alpha*255))).astype('uint8')
image = Image.fromarray(rgba)
bounds = image.getbbox()
image = image.crop((bounds[0]-8, bounds[1]-8, bounds[2]+8, bounds[3]+8))
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
image.save(OUTPUT)
report = dict(source=str(SOURCE.relative_to(MOTION.parent)),
              source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              crop=list(box), excluded_region={'x_min':2160,'y_min':928},
              output_size=list(image.size),
              content='Original red and blue axes, rotation arcs and translation arrows; no formulas or enclosing box.',
              processing='Exact pixel extraction with background transparency. No geometry or colors redrawn.')
OUTPUT.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
