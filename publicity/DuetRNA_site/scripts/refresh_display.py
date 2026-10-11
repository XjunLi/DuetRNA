#!/usr/bin/env python3
"""Restore the static hero and keep all demonstrations in the following section."""
from __future__ import annotations
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parents[1]
REV='20261002-final'
HERO='''<figure class="hero-visual"><span class="hero-label">Base frame</span>
<img class="hero-rna" src="media/rna-illustration.webp" alt="RNA structure illustration with distinct backbone and nucleobases" width="217" height="400" fetchpriority="high">
<span class="hero-label sugar">Sugar frame</span><figcaption>RNA structure illustration · dual-frame geometry</figcaption></figure>'''
CSS='''
/* Static hero and structure thumbnails: 2026-10-02. */
.hero{grid-template-columns:1.25fr 1fr;gap:40px;padding:72px 0 58px;overflow:clip}
.hero h1{font-size:clamp(72px,8.5vw,112px);margin:27px 0 24px}
.hero-description{max-width:540px}.author-strip{padding:26px 0 32px}
.viewer-options{grid-template-columns:repeat(3,minmax(0,1fr));gap:18px;margin:26px 0 0}
.viewer-link{padding:0;overflow:hidden;text-align:left;border-radius:8px;background:var(--white)}
.viewer-link:hover{background:var(--white);border-color:var(--sage)}
.viewer-thumb{position:relative;display:block;aspect-ratio:4/3;background:var(--white)}
.viewer-thumb img{width:100%;height:100%;object-fit:contain}
.viewer-cue{position:absolute;right:13px;bottom:13px;padding:5px 9px;border:1px solid var(--line);border-radius:4px;background:#fffefaee;color:var(--ink);font:14px var(--serif)}
.viewer-meta{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:16px 19px;border-top:1px solid var(--line)}
.viewer-meta b{font:24px var(--serif)}.viewer-meta small{margin:0;font-size:14px}
#explore{padding-top:38px;padding-bottom:48px}#explore h2{font-size:32px}
#explore .lead{font-size:17px;max-width:830px}.featured-links{flex-wrap:wrap}
@media(min-width:1440px){.hero{padding-top:80px;padding-bottom:70px}}
@media(max-width:950px){.hero{gap:16px}.hero h1{font-size:83px}.viewer-options{gap:14px}.viewer-meta{padding:13px}.viewer-meta b{font-size:22px}.viewer-meta small{font-size:12px}}
@media(max-width:680px){.hero{grid-template-columns:1fr;padding:37px 0 24px;gap:6px}.hero h1{font-size:clamp(66px,21vw,86px);margin:21px 0}.hero-visual{margin:15px auto 10px}.author-strip{padding:23px 0}.viewer-options{grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.viewer-meta{padding:11px;gap:7px;align-items:start;flex-direction:column}.viewer-meta b{font-size:21px}.viewer-meta small{font-size:12px}.viewer-cue{right:8px;bottom:8px;font-size:12px;padding:4px 7px}#explore{padding:30px 0 38px}#explore h2{font-size:29px}}
@media(max-width:359px){.viewer-options{grid-template-columns:1fr}.viewer-meta{flex-direction:row;align-items:center}}
'''

def fragment(text):
    return BeautifulSoup(text,'html.parser')

def set_content(tag,text):
    tag.clear()
    for node in list(fragment(text).contents):tag.append(node)

def main():
    poster_rows=json.loads((ROOT/'media/posters/manifest.json').read_text())['posters']
    poster_map={r['source']:r['poster'] for r in poster_rows}
    for path in ['index.html','demo/index.html']:
        file=ROOT/path;soup=BeautifulSoup(file.read_text(),'html.parser')
        if path=='index.html':
            hero=soup.select_one('.hero')
            for node in hero.select('.film-hero,.hero-visual'):node.decompose()
            hero.append(fragment(HERO).figure)
            set_content(hero.select_one('.hero-description'),'Two complementary views of a nucleotide.<br>One model for its sequence and three-dimensional structure.')
            quiet=hero.select_one('.actions .quiet')
            quiet['href']='#films';set_content(quiet,'Explore the demonstrations <span aria-hidden="true">↓</span>')
            films=soup.select_one('#films')
            set_content(films.select_one('.featured-heading p'),'Completed structures at a glance.<br>Click a preview to play the full trajectory.')
            explore=soup.select_one('#explore')
            explore.extract();films.insert_after(explore)
            head=explore.select_one('.section-head')
            if head:head.decompose()
            set_content(explore.select_one('h2'),'Inspect the generated structures.')
            set_content(explore.select_one('.lead'),'Select a structure to open its interactive 3D view. Each thumbnail shows the corresponding structure; drag to rotate, reset the view or download the PDB.')
            for link in soup.select('.viewer-link[data-viewer]'):
                length=parse_qs(urlparse(link['href']).query)['length'][0]
                score=link.select_one('small').get_text(strip=True)
                set_content(link,f'''<span class="viewer-thumb"><img src="media/structures/{length}nt.webp?v={REV}" width="640" height="480" loading="lazy" decoding="async" alt="Generated {length}-nt RNA structure, matching this interactive viewer"><span class="viewer-cue">Explore in 3D ↗</span></span><span class="viewer-meta"><b>{length} nt</b><small>{score}</small></span>''')
                link['aria-label']=f'Open {length}-nt RNA in 3D, {score}'
            nav=soup.select_one('#site-nav')
            link=nav.select_one('a[href="#explore"]');link.extract()
            nav.select_one('a[href="#films"]').insert_after(link)
            set_content(link,'3D structures')
            set_content(nav.select_one('a[href="#films"]'),'Demos')
            soup.select_one('#citation .section-num').string='05'
        for link in soup.select('[data-inline-video]'):
            source=str((ROOT/Path(path).parent/link['href']).resolve().relative_to(ROOT))
            image=link.select_one('img')
            prefix='../' if path.startswith('demo/') else ''
            image['src']=prefix+poster_map[source]
            image['alt']=image.get('alt','').replace('First frame of','Final frame of')
        for text in list(soup.find_all(string=True)):
            if 'First frame on arrival' in text:
                text.replace_with(text.replace('First frame on arrival','Completed structure on arrival'))
        for asset in soup.select('link[rel="stylesheet"],script[src]'):
            key='href' if asset.name=='link' else 'src'
            asset[key]=asset[key].split('?')[0]+'?v='+REV
        soup.html['data-display-revision']=REV
        file.write_text(str(soup))
    css=ROOT/'assets/refinements.css'
    text=css.read_text();marker='/* Static hero and structure thumbnails: 2026-10-02. */'
    text=text.split(marker)[0].rstrip()+'\n'+CSS
    css.write_text(text)
    player=ROOT/'assets/inline-players.js'
    player.write_text(player.read_text().replace('Static first-frame posters','Static final-frame posters'))
    doc=ROOT/'SITE.md'
    text=doc.read_text()
    text=text.replace('actual first decoded frame','actual final decoded frame')
    text=text.replace('The primary 120-nt film is next to the title. The six-film collection precedes the research sections. All 30 gallery entries show their own first frame without loading MP4 data; clicking plays in place. The native dialog is retained for enlarged paper figures and the 3D viewer.',
        'Keep the original static title + RNA illustration hero. The six-film collection and six interactive structure previews follow the author strip, before the research text. All 30 gallery videos use their actual final decoded frame as the poster; clicking plays the unchanged source from time zero. Each 3D thumbnail is rendered from the exact PDB file behind its link, not from a different same-length video sample. The dialog remains for enlarged paper figures and 3D viewing.')
    marker='## Final-frame and static-hero revision'
    text=text.split(marker)[0].rstrip()+'''\n\n## Final-frame and static-hero revision

- Preserve serif typography and the original static hero. Keep demonstrations below it.
- `scripts/build_posters.py` counts decoded frames and extracts exactly `frame_count - 1`; `media/posters/manifest.json` records the source hash and index.
- `scripts/build_structure_previews.py` renders each actual viewer PDB with Mol* 3.2.0. `media/structures/manifest.json` records the PDB hashes. No substitute samples or invented structures are used.
- `scripts/refresh_display.py` applies the scoped HTML/CSS changes and cache-busted asset URLs. It is not a runtime dependency.
- `scripts/check_site.py` checks layout, source provenance, poster loading and real click-to-play behavior. Runtime pages require no build step.
'''
    doc.write_text(text)
    print('Restored static hero; demos grouped next; final-frame posters and six matching 3D thumbnails linked.',flush=True)

if __name__=='__main__':main()
