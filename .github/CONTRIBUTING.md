# Repository layout and maintenance

The Git repository root is the main project folder. Keep all publicity materials together:

```text
DuetRNA/            # Released model, preprocessing, training and evaluation code
publicity/
  DuetRNA_site/      # Project website, media, viewers and maintenance scripts
  DuetRNA_blog/      # Blog and WeChat HTML versions
  DuetRNA_小红书/    # Editable cards, caption and exported draft images
README.md
```

Model code lives in `DuetRNA/`; run its installation, training and evaluation
commands from that directory. Keep datasets, checkpoints, run outputs and
development materials out of the code release. Do not add a `docs/` wrapper
around the website or nest a separate Git repository inside the repository.

GitHub Pages uses `.github/workflows/pages.yml` to publish only `publicity/DuetRNA_site/`. The website URL stays https://xjunli.github.io/DuetRNA/.

Work directly on `main`. Do not create new branches or pull requests unless the owner explicitly changes this instruction. Preserve existing authorship and history; do not force-push. Follow `publicity/DuetRNA_site/SITE_COMMIT_POLICY.md` for commit identity, and `publicity/DuetRNA_site/SITE.md` for website validation.
