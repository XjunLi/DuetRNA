# Demo production source

The approved output is [`../DuetRNA_31s.mp4`](../DuetRNA_31s.mp4) (v27):
30.733 seconds, 1920 × 1080, 30 fps, English captions, no audio.

| Time | Scene |
| --- | --- |
| 0–12.15 s | RNA-to-nucleotide zoom; sugar and base frames appear on the same nucleotide |
| 12.15–25 s | One 110-nt recorded sampling path, with the paper's relative-pose diagram |
| 25–30.733 s | Six generated samples rotate once and fade |

`film.js` renders deterministically through `window.renderFrame(seconds)`.
`film.html` contains the stage and typography, `scatter_motion.mjs` the display
transition, and `timeline.json` the timing and provenance.

## Rebuild

Run from the parent press-kit directory with Node.js 20+, Python, FFmpeg and
Google Chrome installed:

```sh
npm ci --prefix motion
python -m pip install -r requirements.txt
python motion/prepare_gallery_original.py --ffmpeg ffmpeg
python build_demo.py --preview-only --ffmpeg ffmpeg
python build_demo.py --ffmpeg ffmpeg
node motion/serve_preview.cjs --port 8005
```

Use an executable path in place of `ffmpeg` if needed. `FFMPEG_PATH` and
`CHROME_PATH` are also supported by the renderer. `--preview-only` writes diagnostic
keyframes without encoding a new MP4. Generated frames go in `motion/build/` and
are excluded from Git.

The gallery preparation reads six videos in the sibling
`../DuetRNA_site/videos/` directory. Keep the repository directory structure
when rebuilding. Prepared animation geometry is
included in `data/film_data.json`; it can be regenerated with:

```sh
python motion/prep_film.py --skip-gallery --ffmpeg ffmpeg
```

## Scientific provenance

The trajectory contains 101 saved model sampling states from
`sample_32_traj.pdb`. Display interpolation is only between adjacent saved states.
The RNA-to-noise scatter initializes the scene; it is an illustrative transition,
not a reverse model trajectory. The opening nucleotide is residue 36 of the same
RNA. Prepared input coordinates are retained in `../media/sampling/sample_110.json`.

The constraint artwork is extracted from the paper method figure by
`extract_constraint_diagram.py`. It illustrates endpoint-frame relative rotation
and translation, rather than measured errors or a hard lock during sampling.

The endpoint ribbon uses saved C4′ coordinates and base planes. The final six
samples retain their original video colors, with one rotation compressed into
4.6 seconds. Raw inference exports, earlier film versions and review screenshots
are kept locally.
