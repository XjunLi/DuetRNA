# Website maintenance and commit identity

## Branch and publication workflow

The owner's instruction is to work directly on **`main`**. Do not create new branches or pull requests unless the owner explicitly changes this instruction. Pull the latest changes, preserve unrelated work, validate the website and push authorized updates to `main` without force-pushing.

Website files belong in `docs/`; GitHub Pages publishes the `main` branch's `/docs` directory. Keep `.github/workflows/` at the repository root as required by GitHub, with website commands executed from `docs/`.

## Commit identity

For AI-assisted website updates explicitly authorized by the repository owner, publish through the connected GitHub account **XjunLi**, with the account's commit identity **LI JUNZHE**. Do not configure GitHub Actions to commit under `github-actions[bot]`, and do not rename a bot to impersonate the owner.

The three website asset/validation workflows only generate downloadable artifacts and run checks. They use `contents: read`, disable persisted checkout credentials and contain no commit or push step. GitHub Pages may still run as an automated deployment; deployment jobs are separate from source commits.

Review generated assets and test results, then make the source commit through the owner's authorized GitHub connection. Preserve other contributors' attribution and existing commit history. Do not force-push or rewrite historical authorship without a separate explicit request.

For the final-frame refresh, the reviewed browser report contains 14 passing checks. It uses real MP4 playback in Chrome 154.0.8037.57 and real Mol* 3.2.0 renders from the six original viewer PDB payloads. Both pages were checked at nine widths from 320 to 1440 CSS pixels. The static hero, final-frame film posters, serif typography and on-demand media lifecycle are preserved.
