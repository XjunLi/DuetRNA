# DuetRNA press kit

Current materials for sharing and publication. The articles and video are the
approved versions; Word documents are distributed separately.

## Deliverables

| Material | Files |
| --- | --- |
| Chinese article | [Markdown](DuetRNA.md) · [LaTeX Markdown](DuetRNA_LaTeX.md) · [HTML](blog_zh.html) |
| English article | [Markdown](DuetRNA_en.md) · [LaTeX Markdown](DuetRNA_en_LaTeX.md) · [HTML](blog_en.html) |
| Final demo | [DuetRNA_31s.mp4](DuetRNA_31s.mp4) · [SRT captions](DuetRNA_31s.srt) · [Local preview](preview.html) |
| Rednote copy | [posts/Rednote_zh.md](posts/Rednote_zh.md) |
| LinkedIn copy | [posts/LinkedIn_en.md](posts/LinkedIn_en.md) |
| X copy | [posts/X_en.md](posts/X_en.md) |
| Combined platform copy | [social_posts.md](social_posts.md) · [HTML](social_posts.html) |

The demo is English, 1920 × 1080, 30 fps, 30.733 seconds, with no audio.
It shows dual-frame representation, a recorded 110-nt sampling path with the
relative-pose diagram, and six generated structures.

The standard Markdown files use formula images for compatibility. The LaTeX
variants contain the same prose with editable mathematics. Keep `media/` alongside
the Markdown and portable HTML files when copying them. The separately published
interactive pages live in [`../DuetRNA_blog/`](../DuetRNA_blog/).

## Source and local preview

- `media/`: article figures, equations, artwork and prepared 110-nt sampling data.
- [`motion/`](motion/README.md): animation source, prepared geometry and rendering instructions.
- `markdown_manifest.json`: article/figure hashes and source paths relative to this directory.
- `export_blog_markdown.py`: regenerate portable formula-image Markdown from the edited LaTeX version.
- `build_preview.cjs`: render equations and portable HTML previews.
- `sync_publication_blog.py`: update the interactive blog from approved Markdown; review its output before publishing.
- `build_demo.py`: render and package the video.

To view the existing deliverables, run from this directory:

```sh
node motion/serve_preview.cjs --port 8005
```

Open `http://127.0.0.1:8005/preview.html`, `blog_zh.html` or `blog_en.html`.
GitHub's file viewer shows source for HTML; use the published blog links for the
interactive pages.

For article rebuilds, use Node.js 20 or later, Python, and installed Google Chrome:

```sh
npm ci
python -m pip install -r requirements.txt
python export_blog_markdown.py
node build_preview.cjs
```

Set `CHROME_PATH` if Chrome is installed in a custom location. Rebuilds intentionally
write generated files; they are not needed to read or download the final materials.
Old versions, raw inference exports, local audits, dependency folders and rendered
frame caches remain outside the published kit.
