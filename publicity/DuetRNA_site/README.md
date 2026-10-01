# DuetRNA project website

This folder contains the website, media, viewers and site-maintenance scripts.
It sits beside the Blog and Xiaohongshu versions under the repository's `publicity/` folder.
GitHub Actions publishes this folder at [xjunli.github.io/DuetRNA](https://xjunli.github.io/DuetRNA/).

Preview from the repository root:

```sh
python -m http.server 8000 --directory publicity/DuetRNA_site
```

See [`SITE.md`](SITE.md) for editing and validation, and
[`SITE_COMMIT_POLICY.md`](SITE_COMMIT_POLICY.md) for maintenance rules.

Work directly on `main`. Do not create additional branches or pull requests unless the owner explicitly changes this instruction.
