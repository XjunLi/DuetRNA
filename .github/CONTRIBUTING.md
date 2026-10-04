# Repository layout and maintenance

The Git repository root contains the model code. Keep all publicity materials
together, and keep the paper and method figures in `Docs/`:

```text
duetrna/           # Model, preprocessing, training and generation code
benchmark/         # Evaluation protocols and metrics
evaluation_analysis/ # Extended evaluation analyses
external_tools/    # Included evaluation adapters and their source dependencies
publicity/
  DuetRNA_site/      # Project website, media, viewers and maintenance scripts
  DuetRNA_blog/      # Blog and WeChat HTML versions
  DuetRNA_小红书/    # Editable cards, caption and exported draft images
Docs/              # Paper PDF and method figures
pyproject.toml
README.md
```

After cloning, run `cd DuetRNA`. Run installation, training and evaluation
commands from this repository root. Keep datasets, checkpoints, run outputs
and local development materials out of the code release. Track the paper and
method figures in `Docs/`. Do not add a `docs/` wrapper
around the website or nest a separate Git repository inside the repository.

GitHub Pages uses `.github/workflows/pages.yml` to publish only `publicity/DuetRNA_site/`. The website URL stays https://xjunli.github.io/DuetRNA/.

The model code is still under local development. Do not commit, push, or sync
it to the public repository until the owner explicitly authorizes publication.
Keep the existing training objective and configuration when reorganizing code.
A large cohesive module is acceptable; split responsibilities only when that
clarifies dependencies, and keep comments and documentation aligned with behavior.

Keep README edits limited to requested changes and verified usage instructions.
Cite the bioRxiv manuscript supplied by the owner, not the GenBio workshop
version. Use `Code coming soon.` for the pending release status. Do not add process
commentary, agent validation logs, or Markdown flowcharts to the README.

Work directly on `main`. Do not create new branches or pull requests unless the owner explicitly changes this instruction. Preserve existing authorship and history; do not force-push. Follow `publicity/DuetRNA_site/SITE_COMMIT_POLICY.md` for commit identity, and `publicity/DuetRNA_site/SITE.md` for website validation.
