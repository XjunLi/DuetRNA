#!/usr/bin/env python3
"""One-time, idempotent conversion of the original project pages to visible inline films.
Existing research paragraphs, author details and media files are retained.
"""
from pathlib import Path
from bs4 import BeautifulSoup
import re
import html
R = Path(__file__).resolve().parents[1]

def fragment(s):
    return BeautifulSoup(s, 'html.parser').contents[0]

def film(href, title, poster, prefix='', hero=False, eager=False, length=None):
    escaped = html.escape(title)
    length_attr = f' data-length="{length}"' if length else ''
    priority = ' fetchpriority="high"' if hero else ''
    return f'''<figure class="film-card{' film-hero' if hero else ''}" data-film{length_attr}>
<div class="film-stage"><a class="film-preview" href="{href}" data-inline-video aria-label="Play {escaped}">
<img src="{poster}" alt="First frame of {escaped}" width="640" height="640" loading="{'eager' if eager else 'lazy'}" decoding="async"{priority}>
<span class="film-play"><span class="film-play-icon" aria-hidden="true">▶</span><span>Play trajectory</span></span>
</a></div><figcaption class="film-caption"><span><span class="film-title">{escaped}</span><span class="film-subtitle">{'Featured · dual-frame colour reveal' if hero else prefix}</span></span><button class="film-reset" type="button" hidden>Back to preview</button><a class="film-direct" href="{href}" target="_blank" rel="noopener" aria-label="Open {escaped} separately">↗</a></figcaption><p class="film-status" role="status" aria-live="polite"></p></figure>'''

for rel in ['index.html', 'demo/index.html']:
    p = R / rel
    if 'assets/inline-players.js' in p.read_text():
        continue
    soup = BeautifulSoup(p.read_text(), 'html.parser')
    prefix = '../' if rel.startswith('demo') else ''
    soup.head.append(fragment(f'<link rel="stylesheet" href="{prefix}assets/refinements.css?v=20261001-r2">'))
    soup.head.append(fragment(f'<script src="{prefix}assets/inline-players.js?v=20261001-r2" defer></script>'))
    for mark in soup.select('.footer-brand'):
        mark.decompose()
    for a in soup.select('a[data-video]'):
        href = a['href']; stem = Path(href).stem
        kind = 'dual' if '/dual/' in href or href.startswith('dual/') else 'rainbow'
        L, N = re.match(r'(\d+)_na_sample_(\d+)', stem).groups()
        a.replace_with(fragment(film(href, f'{L} nt · sample {N}', f'{prefix}media/posters/{kind}-{stem}.webp',
            prefix='Dual-frame colours' if kind == 'dual' else 'Rainbow rendering', length=a.get('data-length'))))
    if rel == 'index.html':
        soup.select_one('.hero-visual').replace_with(fragment(film('demo/dual/120_na_sample_10.mp4', '120 nt · sample 10', 'media/posters/dual-120_na_sample_10.webp', hero=True, eager=True)))
        hero = soup.select_one('.hero')
        hero.select_one('.hero-description').clear()
        hero.select_one('.hero-description').append('Joint RNA sequence–structure generation through coupled base and sugar frames. Select the preview to watch a complete generation trajectory.')
        last = hero.select('.actions a')[-1]; last['href'] = '#films'; last.clear(); last.append('Watch the trajectories ↓')
        soup.select_one('#site-nav').insert(0, fragment('<a href="#films">Films</a>'))
        explore = soup.select_one('#explore')
        grid = explore.select_one('.sample-grid').extract(); grid['class'] = ['sample-grid', 'film-grid']
        feature = fragment('''<section class="featured-films" id="films"><div class="featured-heading"><div><div class="eyebrow">Generation in view</div><h2>From noise to RNA geometry.</h2></div><p>Six lengths, six trajectories.<br>First frame on arrival. Click to play.</p></div><div class="film-grid-slot"></div><div class="featured-links"><p>Grey → nucleobases in green · sugar–phosphate backbone in blue.</p><a class="inline-link" href="demo/">Browse all 30 videos ↗</a></div></section>''')
        feature.select_one('.film-grid-slot').replace_with(grid)
        soup.select_one('.author-strip').insert_after(feature)
        for el in list(explore.children):
            if getattr(el, 'name', None) in ['div', 'h3']:
                if el.get('class') in [['section-head'], ['viewer-options']]:
                    continue
                el.decompose()
            elif getattr(el, 'name', None) == 'p':
                el.decompose()
        explore.select_one('.section-head').insert_after(fragment('<h2>Inspect the generated structures.</h2>'))
        explore.select_one('h2').insert_after(fragment('<p class="lead">Open a structure in 3D to inspect its geometry. Drag to rotate, reset the view or download the PDB. One viewer runs at a time.</p>'))
    else:
        for grid in soup.select('.sample-grid'):
            grid['class'] = ['sample-grid', 'film-grid']
        pintro = soup.select_one('.page-hero p'); pintro.clear()
        pintro.append('Thirty recorded videos across six sequence lengths. Every preview is the first frame of its own trajectory. Click to play in place; filter below to compare samples of the same length.')
        line = soup.select_one('main > .resource-line p')
        if line:
            line.clear(); line.append('Previews are static first frames. A video starts only when selected; switching samples releases the previous player.')
    p.write_text(str(soup))

p = R / 'viewer/index.html'
s = p.read_text()
if 'refinements.css' not in s:
    p.write_text(s.replace('</head>', '<link rel="stylesheet" href="../assets/refinements.css?v=20261001-r2"></head>'))
p = R / 'assets/site.css'
s = p.read_text().replace("--sans:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif", '--text:var(--serif)').replace('var(--sans)', 'var(--serif)')
p.write_text(s)
p = R / 'SITE.md'; text = p.read_text()
if '## Serif typography and visible films' not in text:
    text += '''

## Serif typography and visible films

The project owner's typography preference is serif throughout: Georgia with Palatino/Songti/serif fallbacks, including body, navigation, buttons, labels and the viewer UI. Do not reintroduce system-ui, Segoe UI, Inter or other sans-serif interface defaults. Keep BibTeX in monospace. Do not invent laboratory logos; show the laboratory's actual name in plain text.

- `assets/refinements.css` contains the serif typography and poster-first film layouts.
- `assets/inline-players.js` implements in-place playback and releases the previous video when another starts. Videos pause off screen or when the document is hidden. No automatic resume or simultaneous playback.
- `media/posters/` contains the actual first decoded frame of every sample video, encoded as small WebP files. `manifest.json` records the source, frame index, duration and byte size.
- `scripts/build_posters.py` regenerates posters with FFmpeg; the Build static video previews workflow can be invoked manually after video changes.
- `scripts/check_site.py` checks real poster loading, real MP4 playback, resource cleanup, filtering and responsive layouts using Playwright.
- `scripts/apply_editorial_revision.py` is an idempotent migration of the previous page layout. It is not a runtime or publication dependency. After migration, edit the generated HTML files directly.

The primary 120-nt film is next to the title. The six-film collection precedes the research sections. All 30 gallery entries show their own first frame without loading MP4 data; clicking plays in place. The native dialog is retained for enlarged paper figures and the 3D viewer.
'''
    text = text.replace('A click creates one media node inside a shared native dialog.', 'A video click creates one inline player; figure and 3D-viewer clicks use the native dialog.')
    p.write_text(text)
print('Editorial revision applied; existing migrated pages are left intact on reruns.')
