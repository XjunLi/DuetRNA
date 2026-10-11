#!/usr/bin/env python3
"""Export the visible HTML cards to PNG, with a contact sheet and layout checks.

Usage:
    python export_cards.py
    python export_cards.py --scale 2
    python export_cards.py --browser /path/to/chrome

Requires Python 3.10+, Playwright, and Pillow. No remote rendering service is used.
"""
from __future__ import annotations

import argparse
import json
import base64
import mimetypes
import sys
from pathlib import Path

try:
    from bs4 import BeautifulSoup
    from PIL import Image, ImageOps
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright
except ImportError as exc:
    raise SystemExit(
        "缺少依赖。请先运行：python -m pip install -r requirements.txt"
    ) from exc

ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1080, 1440

LAYOUT_CHECK = """() => Array.from(document.querySelectorAll('article.card')).map(card => {
  const box = card.getBoundingClientRect();
  const main = card.querySelector('.card-main').getBoundingClientRect();
  const source = card.querySelector('.source')?.getBoundingClientRect();
  const footer = card.querySelector('.footer').getBoundingClientRect();
  const brokenImages = Array.from(card.querySelectorAll('img')).filter(
    im => !im.complete || im.naturalWidth === 0
  ).map(im => im.getAttribute('src'));
  const textOverflow = [];
  const walker = document.createTreeWalker(card, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    const node = walker.currentNode;
    if (!node.textContent.trim()) continue;
    const range = document.createRange(); range.selectNodeContents(node);
    for (const r of range.getClientRects()) {
      if (r.left < box.left - 1 || r.right > box.right + 1 ||
          r.top < box.top - 1 || r.bottom > box.bottom + 1) {
        textOverflow.push(node.textContent.trim().slice(0, 70)); break;
      }
    }
  }
  return {
    id: card.id, width: Math.round(box.width), height: Math.round(box.height),
    mainToSourceGap: source ? Math.round(source.top - main.bottom) : null,
    sourceToFooterGap: source ? Math.round(footer.top - source.bottom) : null,
    mainToFooterGap: Math.round(footer.top - main.bottom),
    brokenImages, textOverflow
  };
})"""


def inline_document(html: Path) -> str:
    """Inline this package's local CSS/images for deterministic, offline rendering."""
    soup = BeautifulSoup(html.read_text(encoding="utf-8"), "html.parser")
    for link in soup.select('link[rel="stylesheet"]'):
        href = link.get("href", "")
        if "://" in href or href.startswith("//"):
            raise ValueError("导出需要本地样式文件，请把远程 CSS 保存到源码目录。")
        path = (html.parent / href).resolve()
        style = soup.new_tag("style")
        style.string = path.read_text(encoding="utf-8")
        link.replace_with(style)
    for image in soup.select("img"):
        src = image.get("src", "")
        if src.startswith("data:"):
            continue
        if "://" in src or src.startswith("//"):
            raise ValueError("导出需要本地图片，请把远程素材保存到 assets 目录。")
        path = (html.parent / src).resolve()
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        image["src"] = f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")
    return str(soup)


def contact_sheet(paths: list[Path], destination: Path) -> None:
    """Four columns and as many rows as needed for the visible cards."""
    thumb_w, thumb_h, gap, columns = 270, 360, 16, 4
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new(
        "RGB", (columns * thumb_w + (columns + 1) * gap,
                rows * thumb_h + (rows + 1) * gap), "#e8ebe2"
    )
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            thumbnail = ImageOps.contain(
                image.convert("RGB"), (thumb_w, thumb_h), Image.Resampling.LANCZOS
            )
            row = index // columns
            row_count = min(columns, len(paths) - row * columns)
            row_offset = (columns - row_count) * (thumb_w + gap) // 2
            x = gap + row_offset + (index % columns) * (thumb_w + gap)
            y = gap + row * (thumb_h + gap)
            sheet.paste(thumbnail, (x, y))
    sheet.save(destination, quality=94, subsampling=0)


def export(html: Path, output: Path, scale: float,
           browser_path: Path | None, check_only: bool) -> None:
    if not html.is_file():
        raise ValueError(f"找不到 HTML：{html}")
    if browser_path is not None and not browser_path.is_file():
        raise ValueError(f"找不到浏览器：{browser_path}")
    output.mkdir(parents=True, exist_ok=True)
    launch_options: dict = {"headless": True}
    if browser_path is not None:
        launch_options["executable_path"] = str(browser_path)
    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(**launch_options)
        except PlaywrightError as exc:
            raise RuntimeError(
                "无法启动 Chromium。请运行：python -m playwright install chromium\n"
                "已有 Chrome 时也可用 --browser 指定其可执行文件。\n" + str(exc)
            ) from exc
        try:
            page = browser.new_page(
                viewport={"width": 1120, "height": 1500},
                device_scale_factor=scale,
                locale="zh-CN", color_scheme="light", reduced_motion="reduce"
            )
            page_errors: list[str] = []
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            page.set_content(inline_document(html), wait_until="load", timeout=60000)
            page.evaluate("() => document.fonts.ready")
            page.wait_for_function(
                "Array.from(document.images).every(im => im.complete)", timeout=30000
            )
            cards = page.locator("article.card")
            card_count = cards.count()
            if card_count == 0:
                raise RuntimeError("未找到可展示的图卡。")
            report = page.evaluate(LAYOUT_CHECK)
            problems = []
            for card in report:
                if (card["width"], card["height"]) != (WIDTH, HEIGHT):
                    problems.append(f'{card["id"]}: 画布尺寸不是 1080 × 1440')
                if card["mainToSourceGap"] is not None and card["mainToSourceGap"] < 0:
                    problems.append(f'{card["id"]}: 正文与来源脚注重叠')
                if card["sourceToFooterGap"] is not None and card["sourceToFooterGap"] < 0:
                    problems.append(f'{card["id"]}: 来源脚注与页脚重叠')
                if card["mainToFooterGap"] < 0:
                    problems.append(f'{card["id"]}: 正文与页脚重叠')
                if card["brokenImages"]:
                    problems.append(f'{card["id"]}: 图片加载失败')
                if card["textOverflow"]:
                    problems.append(f'{card["id"]}: 文字超出画布')
            result = {"html": html.name, "scale": scale, "card_count": card_count, "cards": report,
                      "page_errors": page_errors, "problems": problems}
            (output / "layout-report.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            if problems or page_errors:
                raise RuntimeError("排版检查未通过：\n" + "\n".join(problems + page_errors))
            if check_only:
                print(f"{card_count} 张图卡排版检查通过。")
                return
            exported: list[Path] = []
            for index in range(cards.count()):
                target = output / f"{index + 1:02d}.png"
                cards.nth(index).screenshot(path=str(target), animations="disabled")
                with Image.open(target) as image:
                    expected = (round(WIDTH * scale), round(HEIGHT * scale))
                    if image.size != expected:
                        raise RuntimeError(f"导出尺寸错误：{target.name}: {image.size}")
                exported.append(target)
                print(f"已导出 {target.name}")
            contact_sheet(exported, output / "preview.jpg")
            print(f"完成：{output}\n已保存 {card_count} 张 PNG、总览 preview.jpg 和排版检查记录。")
        finally:
            browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, default=ROOT / "cards.html")
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    parser.add_argument("--scale", type=float, default=1,
                        help="像素倍率：1=1080×1440，2=2160×2880。")
    parser.add_argument("--browser", type=Path, help="可选：Chrome/Chromium 可执行文件。")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not 0.5 <= args.scale <= 4:
        parser.error("--scale 必须在 0.5 到 4 之间。")
    try:
        export(args.html.expanduser().resolve(), args.output.expanduser().resolve(),
               args.scale, args.browser.expanduser().resolve() if args.browser else None,
               args.check_only)
    except (ValueError, RuntimeError, PlaywrightError, OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
