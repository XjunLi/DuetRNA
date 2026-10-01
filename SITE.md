# DuetRNA project website

A static GitHub Pages site. No build system, npm dependencies or external fonts are required.

## Edit

- `index.html`: project text, figures, authors, results and featured sample links.
- `demo/index.html`: all 30 video links and length filters.
- `assets/site.css`: shared palette, typography, layouts and responsive styles.
- `assets/site.js`: menu, reading progress, media dialog, filters and citation copying.
- `viewer/index.html` and `viewer/viewer.js`: one on-demand Mol* viewer.

The original `media/`, `videos/`, `demo/dual/` and six `viewer/*_genonly.html` structure resources are preserved. The new viewer reads the embedded PDB from those legacy files as text; it does not execute their HTML. Keep those files available, including `120nt_s16_genonly.html`.

## Preview

From the repository root:

```sh
python -m http.server 8000
```

Open `http://localhost:8000/` and `/demo/`. The 3D viewer needs network access to the pinned Mol* 3.2.0 assets. It tries jsDelivr, then unpkg. Structure download remains available when the PDB loads but the renderer does not.

## Media lifecycle

No MP4 source, iframe or WebGL renderer is created at initial page load. A click creates one media node inside a shared native dialog. Closing the dialog releases the video source or removes the iframe. Video pauses when the document is hidden. The 3D viewer stops its animation loop when hidden and disposes its plugin on page exit. There is no automatically advancing video carousel.

Research videos and data are unchanged. The small `media/rna-illustration.webp` is a compressed version of the author-provided RNA illustration, not a generated sample or benchmark result. The paper figures remain original PNG resources.

## Validation for the 2026-10-01 refresh

- JavaScript syntax checked with `node --check`.
- Both page layouts rendered at 320, 360, 390, 768, 1024 and 1440 CSS pixels; no document-level horizontal overflow or page-script exceptions in offline Chromium.
- Eleven offline interaction/lifecycle checks cover menu, figures, media cleanup, visibility handling, clipboard fallback, six length filters and PDB extraction/error handling.
- Media API and Mol* tests use fixtures. This is not a live-CDN, real-MP4 decoding or actual-WebGL benchmark. Test those integrations on the deployed site before attributing device-specific speedups.

## Manual release check

Open a real video, close it, then open a different sample; only one should play. Open a 3D structure and test rotation, Reset, PDB download, Escape and reopening. Repeat on Safari and a mobile browser. Verify Paper and Repository links. Disable JavaScript: the research and direct media links should remain readable.
