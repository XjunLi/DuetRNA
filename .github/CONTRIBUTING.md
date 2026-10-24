# Repository layout and maintenance

This public repository contains the model and evaluation code. Keep the paper
and method figures in `Docs/`. Website and outreach sources are maintained in
the private companion repository and are not committed here:

```text
duetrna/           # Model, preprocessing, training and generation code
benchmark/         # Evaluation protocols and metrics
evaluation_analysis/ # Extended evaluation analyses
external_tools/    # Included evaluation adapters and their source dependencies
Docs/              # Paper PDF and method figures
pyproject.toml
README.md
```

After cloning, run `cd DuetRNA`. Run installation, training and evaluation
commands from this repository root. Keep datasets, checkpoints, run outputs
and local development materials out of the code release. Track the paper and
method figures in `Docs/`. Do not nest a separate Git repository inside this
repository.

GitHub Pages uses `.github/workflows/pages.yml` to check out
`XjunLi/DuetRNA-private` read-only, stage its publication sources and deploy the
resulting Pages artifact. The website URL remains
https://xjunli.github.io/DuetRNA/, with Chinese and English blogs at
`/DuetRNA/blog/zh/` and `/DuetRNA/blog/en/`, and image cards at
`/DuetRNA/xiaohongshu/`. Website validation and asset-generation workflows live
with the private sources. Actions deploys the website without committing
generated files or changing source history.

The model code is still under local development. Do not commit, push, or sync
it to the public repository until the owner explicitly authorizes publication.
Keep the existing training objective and configuration when reorganizing code.
A large cohesive module is acceptable; split responsibilities only when that
clarifies dependencies, and keep comments and documentation aligned with behavior.

Keep README edits limited to requested changes and verified usage instructions.
Cite the bioRxiv manuscript supplied by the owner, not the GenBio workshop
version. Use `Code coming soon.` for the pending release status. Do not add process
commentary, agent validation logs, or Markdown flowcharts to the README.

Work directly on `main`. Do not create new branches or pull requests unless the
owner explicitly changes this instruction. Preserve existing authorship and
history; do not force-push. For owner-authorized AI-assisted maintenance, commit
through **XjunLi** as **LI JUNZHE**
(`125676363+XjunLi@users.noreply.github.com`), without bot authorship or
co-authorship. Do not add commit or push steps to Actions. Website-specific
maintenance and validation instructions remain with the private website sources.
