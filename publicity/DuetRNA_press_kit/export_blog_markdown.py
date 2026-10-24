"""Export bilingual research stories or synchronize edited long-form Markdown.

Run in the DuetRNA repository. Requires BeautifulSoup. Formula PNGs are
rendered by build_preview.cjs. For edited_longform manifests, *_LaTeX.md is
the canonical prose; HTML extraction is skipped to preserve editorial changes.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup, NavigableString, Tag

ROOT = Path(__file__).resolve().parents[2]
KIT = Path(__file__).resolve().parent
SOURCE = ROOT / 'publicity/DuetRNA_blog'
SOURCES = {'zh': 'DuetRNA_互动研究手记_v2.html',
           'en': 'DuetRNA_Interactive_Research_Notes_EN.html'}


def tidy(value):
    return re.sub(r'\s+', ' ', value).strip()


class Exporter:
    def __init__(self, lang, output):
        self.lang = lang
        self.output = output
        self.source = SOURCE / SOURCES[lang]
        self.soup = BeautifulSoup(self.source.read_text(encoding='utf-8'), 'html.parser')
        self.base = f'https://xjunli.github.io/DuetRNA/blog/{lang}/'
        self.refs = {li['id']: li.select_one('div > a')['href']
                     for li in self.soup.select('.sources > li')}
        self.equations = []
        self.figures = []
        self.covered = set()

    def inline(self, node):
        if isinstance(node, NavigableString):
            return re.sub(r'\s+', ' ', str(node))
        if not isinstance(node, Tag):
            return ''
        cls = set(node.get('class', []))
        if node.name in ['script', 'style', 'button'] or cls & {'zoom-label', 'ref-no'}:
            return ''
        if node.name == 'br':
            return ' '
        value = ''.join(self.inline(c) for c in node.children)
        if node.name in ['strong', 'b']:
            return ' **' + value.strip() + '** '
        if node.name in ['em', 'i']:
            return ' *' + value.strip() + '* '
        if node.name == 'code':
            return '`' + value.strip() + '`'
        if node.name == 'a':
            target = node.get('href', '')
            if target.startswith('#ref-'):
                target = self.refs[target[1:]]
                return ' [[' + value.strip() + ']](' + target + ') '
            elif target.startswith('#'):
                target = self.base + target
            else:
                target = urljoin(self.base, target)
            return '[' + value.replace('↗', '').strip() + '](' + target + ')'
        if node.name == 'sup' and not node.find('a'):
            return '<sup>' + value + '</sup>'
        if node.name == 'sub':
            return '<sub>' + value + '</sub>'
        return value

    def plain(self, node):
        return tidy(self.inline(node))

    def table(self, headers, rows):
        def row(cells):
            return '| ' + ' | '.join(c.replace('|', '\\|').replace('\n', ' ') for c in cells) + ' |'
        return row(headers) + '\n' + row(['---']*len(headers)) + '\n' + '\n'.join(row(r) for r in rows) + '\n\n'

    def block(self, node):
        if not isinstance(node, Tag):
            return ''
        cls = set(node.get('class', []))
        if node.name in ['script', 'style', 'button'] or cls & {'section-head', 'signature', 'stat-strip'}:
            return ''
        if node.get('id') == 'flow-demo':
            label = node.select_one('h3').get_text(' ', strip=True)
            return f'[{label}]({self.base}#flow-demo)\n\n'
        if node.name == 'figure':
            img = node.find('img')
            number = len(self.figures) + 1
            path = f'media/blog_figure_{number:02}.png'
            data = base64.b64decode(img['src'].split(',', 1)[1])
            target = self.output / path
            if target.exists():
                assert target.read_bytes() == data, f'Language images differ: {path}'
            else:
                target.write_bytes(data)
            caption = self.plain(node.find('figcaption'))
            caption = re.sub(r'\s*\[.*?(?:大图|full size|Enlarge).*?\]\([^)]*\)', '', caption, flags=re.I)
            caption = caption.replace('点击查看大图 ↗', '').replace('Click to enlarge ↗', '').strip()
            alt = img.get('alt', '').rstrip('1234567890').strip().replace('[', '').replace(']', '')
            self.figures.append({'file': path, 'sha256': hashlib.sha256(data).hexdigest()})
            return f'![{alt}]({path})\n\n{caption}\n\n'
        if 'math-formula' in cls:
            tex = node['data-latex'].strip()
            self.equations.append(tex)
            return '\n$$\n' + tex + '\n$$\n\n'
        if node.name == 'table':
            rows = [[self.plain(c) for c in tr.find_all(['td', 'th'], recursive=False)] for tr in node.find_all('tr')]
            return self.table(rows[0], rows[1:])
        if 'duo-explain' in cls:
            rows = [[r.find('h3').get_text(' ', strip=True), self.plain(r.find('p'))] for r in node.select('.role')]
            return self.table(['几何职责', '描述什么'] if self.lang == 'zh' else ['Geometric role', 'What it describes'], rows)
        if 'evidence-visual' in cls:
            title = node.select_one('.evidence-head').get_text(' ', strip=True)
            rows = [[self.plain(r.find('span')), self.plain(r.find('b'))] for r in node.select('.evidence-row')]
            return '**' + title + '**\n\n' + self.table(['坐标系', '旋转漂移'] if self.lang == 'zh' else ['Frame', 'Rotation drift'], rows)
        if 'benchmark-pair' in cls:
            result = ''
            for item in node.select('.benchmark'):
                title = self.plain(item.select_one('.bench-tag'))
                rows = [[self.plain(r.find('span')), self.plain(r.find('b'))] for r in item.select('.bench-bar-row')]
                result += '### ' + title + '\n\n'
                result += self.table(['方法', '计算自洽率'] if self.lang == 'zh' else ['Method', 'Self-consistency'], rows)
                gain = item.select_one('.gain')
                self.covered.add(id(gain))
                result += self.plain(gain) + '。\n\n' if self.lang == 'zh' else self.plain(gain) + '.\n\n'
                result += self.plain(item.find('small')) + '\n\n'
            return result
        if 'ablation' in cls:
            rows = [[self.plain(r.find('span')), self.plain(r.find('b'))] for r in node.select('.ablation-row')]
            note = node.select_one('.chart-note')
            self.covered.add(id(note))
            return self.table(['模型', 'IF / RhoFold'] if self.lang == 'zh' else ['Model', 'IF / RhoFold'], rows) + self.plain(note) + '\n\n'
        if 'output-ribbon' in cls:
            return ' → '.join(self.plain(d.find('span')) + ': ' + self.plain(d.find('b')) for d in node.find_all('div', recursive=False)) + '\n\n'
        if 'resource-links' in cls:
            return ' · '.join(self.plain(a) for a in node.find_all('a')) + '\n\n'
        if 'source-disclosure' in cls:
            out = '### ' + ('参考资料与数据出处' if self.lang == 'zh' else 'References and data sources') + '\n\n'
            for index, item in enumerate(node.select('.sources > li'), 1):
                div = item.find('div')
                paragraph = div.find('p')
                self.covered.add(id(paragraph))
                out += f'{index}. {self.plain(div.find("a"))}. {self.plain(paragraph)}\n'
            return out + '\n'
        if node.name in ['h1', 'h2', 'h3', 'h4']:
            return '#' * int(node.name[1]) + ' ' + self.plain(node) + '\n\n'
        if node.name == 'p':
            self.covered.add(id(node))
            return self.plain(node) + '\n\n'
        if node.name == 'blockquote' or 'reflection-card' in cls:
            body = ''.join(self.block(c) for c in node.children).strip()
            if not body:
                body = self.plain(node)
            return '\n'.join('> '+line if line else '>' for line in body.splitlines()) + '\n\n'
        if node.name in ['ul', 'ol']:
            return ''.join('- '+self.plain(li)+'\n' for li in node.find_all('li', recursive=False))+'\n\n'
        return ''.join(self.block(c) for c in node.children)

    def export(self):
        hero = self.soup.select_one('.hero-copy')
        heading = self.plain(hero.find('h1'))
        author = hero.select_one('.byline strong').get_text(' ', strip=True)
        affiliation = hero.select_one('.byline span').get_text(' ', strip=True)
        raw = '# ' + heading + '\n\n' + self.plain(hero.select_one('.hero-deck')) + '\n\n'
        raw += f'**{author}**  \n{affiliation}\n\n'
        raw += f'[{"在线博客" if self.lang == "zh" else "Online research story"}]({self.base}) · '
        raw += '[DuetRNA](https://www.biorxiv.org/content/10.64898/2026.08.18.745543v1)\n\n'
        raw += self.block(self.soup.select_one('main.article'))
        raw = re.sub(r'\n{3,}', '\n\n', raw).strip()+'\n'
        missing = []
        for p in self.soup.select('main.article p'):
            if p.find_parent(id='flow-demo') or p.find_parent(class_='duo-explain'):
                continue
            if id(p) not in self.covered:
                missing.append(p.get_text(' ', strip=True))
        assert not missing, missing
        assert len(self.equations) == 7 and len(self.figures) == 3
        portable = raw
        for i, tex in enumerate(self.equations, 1):
            label = f'公式 {i}' if self.lang == 'zh' else f'Equation {i}'
            portable = portable.replace('$$\n'+tex+'\n$$', f'![{label}](media/equation_{i:02}.png)', 1)
        stem = 'DuetRNA' if self.lang == 'zh' else 'DuetRNA_en'
        (self.output/f'{stem}.md').write_text(portable, encoding='utf-8')
        (self.output/f'{stem}_LaTeX.md').write_text(raw, encoding='utf-8')
        return {'language': self.lang, 'source': Path(os.path.relpath(self.source, self.output)).as_posix(),
                'source_sha256': hashlib.sha256(self.source.read_bytes()).hexdigest(),
                'body_paragraphs_checked': len(self.covered), 'missing_paragraphs': missing,
                'equations': self.equations, 'figures': self.figures,
                'markdown': f'{stem}.md', 'latex_markdown': f'{stem}_LaTeX.md'}


def sync_longform(output, reports):
    """Keep edited long-form prose; regenerate only the portable formula variant."""
    for report in reports:
        raw = (output / report['latex_markdown']).read_text(encoding='utf-8')
        equations = re.findall(r'\$\$\s*\n(.*?)\n\$\$', raw, re.S)
        if equations != report['equations']:
            raise ValueError('Formula changes require updated PNG assets and equation metadata.')
        portable = raw
        for index, tex in enumerate(equations, 1):
            label = ('公式 ' if report['language'] == 'zh' else 'Equation ') + str(index)
            portable = portable.replace('$$\n' + tex + '\n$$',
                                        f'![{label}](media/equation_{index:02}.png)', 1)
        (output / report['markdown']).write_text(portable, encoding='utf-8')
        report['markdown_sha256'] = hashlib.sha256(portable.encode('utf-8')).hexdigest()
        report['latex_markdown_sha256'] = hashlib.sha256(raw.encode('utf-8')).hexdigest()
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=KIT)
    args = parser.parse_args()
    (args.output_dir/'media').mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / 'markdown_manifest.json'
    previous = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else []
    if previous and all(r.get('content_mode') == 'edited_longform' for r in previous):
        reports = sync_longform(args.output_dir, previous)
    else:
        reports = [Exporter(lang, args.output_dir).export() for lang in ['zh', 'en']]
    assert reports[0]['equations'] == reports[1]['equations']
    (args.output_dir/'markdown_manifest.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps([{k:v for k,v in r.items() if k not in ['equations','figures']} for r in reports], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
