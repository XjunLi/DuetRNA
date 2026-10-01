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

No MP4 source, iframe or WebGL renderer is created at initial page load. A video click creates one inline player; figure and 3D-viewer clicks use the native dialog. Closing the dialog releases the video source or removes the iframe. Video pauses when the document is hidden. The 3D viewer stops its animation loop when hidden and disposes its plugin on page exit. There is no automatically advancing video carousel.

Research videos and data are unchanged. The small `media/rna-illustration.webp` is a compressed version of the author-provided RNA illustration, not a generated sample or benchmark result. The paper figures remain original PNG resources.

## Validation for the 2026-10-01 refresh

- JavaScript syntax checked with `node --check`.
- Both page layouts rendered at 320, 360, 390, 768, 1024 and 1440 CSS pixels; no document-level horizontal overflow or page-script exceptions in offline Chromium.
- Eleven offline interaction/lifecycle checks cover menu, figures, media cleanup, visibility handling, clipboard fallback, six length filters and PDB extraction/error handling.
- Media API and Mol* tests use fixtures. This is not a live-CDN, real-MP4 decoding or actual-WebGL benchmark. Test those integrations on the deployed site before attributing device-specific speedups.

## Manual release check

Open a real video, close it, then open a different sample; only one should play. Open a 3D structure and test rotation, Reset, PDB download, Escape and reopening. Repeat on Safari and a mobile browser. Verify Paper and Repository links. Disable JavaScript: the research and direct media links should remain readable.


## Serif typography and visible films

The project owner's typography preference is serif throughout: Georgia with Palatino/Songti/serif fallbacks, including body, navigation, buttons, labels and the viewer UI. Do not reintroduce system-ui, Segoe UI, Inter or other sans-serif interface defaults. Keep BibTeX in monospace. Do not invent laboratory logos; show the laboratory's actual name in plain text.

- `assets/refinements.css` contains the serif typography and poster-first film layouts.
- `assets/inline-players.js` implements in-place playback and releases the previous video when another starts. Videos pause off screen or when the document is hidden. No automatic resume or simultaneous playback.
- `media/posters/` contains the actual first decoded frame of every sample video, encoded as small WebP files. `manifest.json` records the source, frame index, duration and byte size.
- `scripts/build_posters.py` regenerates posters with FFmpeg; the Build static video previews workflow can be invoked manually after video changes.
- `scripts/check_site.py` checks real poster loading, real MP4 playback, resource cleanup, filtering and responsive layouts using Playwright.
- `scripts/apply_editorial_revision.py` is an idempotent migration of the previous page layout. It is not a runtime or publication dependency. After migration, edit the generated HTML files directly.

The primary 120-nt film is next to the title. The six-film collection precedes the research sections. All 30 gallery entries show their own first frame without loading MP4 data; clicking plays in place. The native dialog is retained for enlarged paper figures and the 3D viewer.
