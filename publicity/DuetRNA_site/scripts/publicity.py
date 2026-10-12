#!/usr/bin/env python3
"""Preview or stage the three publicity formats while keeping their sources separate."""
from __future__ import annotations

import argparse
import functools
import http.server
import io
import re
import posixpath
import shutil
from pathlib import Path
from urllib.parse import unquote, urlsplit

SITE = Path(__file__).resolve().parents[1]
PUBLICITY = SITE.parent
BLOGS = {
    'zh': PUBLICITY / 'DuetRNA_blog' / 'DuetRNA_互动研究手记_v2.html',
    'en': PUBLICITY / 'DuetRNA_blog' / 'DuetRNA_Interactive_Research_Notes_EN.html',
}
CARDS = PUBLICITY / 'DuetRNA_小红书'


def render_cards() -> bytes:
    """Keep the page and its current styles in one response, avoiding stale CSS."""
    source = (CARDS / 'cards.html').read_text('utf-8')
    style = (CARDS / 'cards.css').read_text('utf-8')
    source, count = re.subn(r'<link\b[^>]*href="cards\.css(?:\?[^"]*)?"[^>]*>',
                            lambda match: '<style id="card-styles">' + style + '</style>', source, count=1)
    if count != 1:
        raise ValueError('The card stylesheet link was not found.')
    return source.encode('utf-8')


def render_blog(language: str) -> bytes:
    """Resolve language links for /blog/<language>/ without changing either source."""
    source = BLOGS[language].read_text('utf-8')
    for code, path in BLOGS.items():
        source = source.replace('href="' + path.name + '"', 'href="../' + code + '/"')
    return source.encode('utf-8')


def build(destination: Path) -> None:
    destination = destination.resolve()
    if destination.exists():
        raise ValueError('Choose a new output directory; existing files are never replaced.')
    if destination.is_relative_to(PUBLICITY):
        raise ValueError('The output must be outside the publicity source tree.')
    shutil.copytree(SITE, destination, ignore=shutil.ignore_patterns(
        'scripts', 'qa', '__pycache__', 'README.md', 'SITE.md', 'SITE_COMMIT_POLICY.md'))
    for language in BLOGS:
        blog_dir = destination / 'blog' / language
        blog_dir.mkdir(parents=True)
        (blog_dir / 'index.html').write_bytes(render_blog(language))
    cards_dir = destination / 'xiaohongshu'
    cards_dir.mkdir()
    rendered_cards = render_cards()
    (cards_dir / 'index.html').write_bytes(rendered_cards)
    (cards_dir / 'cards.html').write_bytes(rendered_cards)
    shutil.copy2(CARDS / 'cards.css', cards_dir / 'cards.css')
    shutil.copytree(CARDS / 'assets', cards_dir / 'assets')
    print(f'Pages artifact: {destination}')


class Preview(http.server.SimpleHTTPRequestHandler):
    """The preview resolves the published routes directly to the editable sources."""

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def send_head(self):
        path = Path(self.translate_path(self.path))
        if path == CARDS / 'cards.html':
            data = render_cards()
        elif path in BLOGS.values():
            language = next(code for code, document in BLOGS.items() if path == document)
            data = render_blog(language)
        else:
            return super().send_head()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        return io.BytesIO(data)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path in ('/blog/zh', '/blog/en', '/xiaohongshu'):
            self.send_response(301)
            self.send_header('Location', path + '/')
            self.end_headers()
            return
        super().do_GET()

    def translate_path(self, path):
        decoded = unquote(urlsplit(path).path)
        if '\\' in decoded or '\x00' in decoded:
            return str(SITE / '__not_published__')
        normalized = posixpath.normpath(decoded)
        if self.directory == str(CARDS):
            if normalized in ('/', '/index.html'):
                return str(CARDS / 'cards.html')
            return super().translate_path(path)
        for language, document in BLOGS.items():
            if normalized in (f'/blog/{language}', f'/blog/{language}/index.html'):
                return str(document)
        if normalized in ('/xiaohongshu', '/xiaohongshu/index.html', '/xiaohongshu/cards.html'):
            return str(CARDS / 'cards.html')
        if normalized == '/xiaohongshu/cards.css':
            return str(CARDS / 'cards.css')
        if normalized.startswith('/xiaohongshu/assets/'):
            asset = (CARDS / normalized.removeprefix('/xiaohongshu/')).resolve()
            if asset.is_relative_to(CARDS / 'assets'):
                return str(asset)
        if normalized.startswith(('/xiaohongshu/', '/blog/')):
            return str(SITE / '__not_published__')
        return super().translate_path(path)


class BlogPreview(http.server.BaseHTTPRequestHandler):
    """Serve only the publication document, without editing APIs or local drafts."""

    def __init__(self, *args, language: str, **kwargs):
        self.language = language
        super().__init__(*args, **kwargs)

    def document(self, include_body: bool):
        document = BLOGS[self.language]
        path = unquote(urlsplit(self.path).path)
        if path not in ('/', '/index.html', '/' + document.name):
            self.send_error(404)
            return
        source = render_blog(self.language).decode('utf-8')
        for code, port in (('zh', 8002), ('en', 8004)):
            source = source.replace('href="../' + code + '/"',
                                    'href="http://127.0.0.1:' + str(port) + '/"')
        data = source.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        if include_body:
            self.wfile.write(data)

    def do_GET(self):
        self.document(True)

    def do_HEAD(self):
        self.document(False)

    def do_POST(self):
        self.send_error(405, 'Read-only publication preview')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    stage = modes.add_parser('build')
    stage.add_argument('--output', type=Path, required=True)
    serve = modes.add_parser('serve')
    serve.add_argument('--port', type=int, default=8003)
    formats = serve.add_mutually_exclusive_group()
    formats.add_argument('--cards-only', action='store_true')
    formats.add_argument('--blog-only', choices=tuple(BLOGS))
    args = parser.parse_args()
    if args.mode == 'build':
        build(args.output)
    else:
        if args.blog_only:
            handler = functools.partial(BlogPreview, language=args.blog_only)
        else:
            handler = functools.partial(Preview, directory=str(CARDS if args.cards_only else SITE))
        with http.server.ThreadingHTTPServer(('127.0.0.1', args.port), handler) as server:
            print(f'Publicity preview: http://127.0.0.1:{args.port}/', flush=True)
            server.serve_forever()


if __name__ == '__main__':
    main()
