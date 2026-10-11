# DuetRNA · 小红书图卡源码

图卡的可编辑 HTML/CSS 源码。当前展示 7 张：原第 2 张研究历史暂时收起，页码已连续重排。保留原版配色、插图与数据图，更新封面、页眉和署名，详细论文出处统一收在 SOURCES.md。

## 文件

- `cards.html`：图卡的文字、图片引用和页面结构。展示中的 ID 为 `card-1`、`card-3` 至 `card-8`；沿用原 ID 以保留逐页样式。原 `card-2` 保存在不渲染的 `template#omitted-history` 中。
- `DuetRNA_小红书八图.html`：兼容原文件名的独立预览，内嵌 CSS 和图片；当前同样为 7 张。
- `cards.css`：完整样式，包括原有的逐页排版微调。
- `assets/`：五张原版 PNG 素材及作者新提供的双坐标系流匹配图。原图字节保留。
- `小红书配文.txt`：标题、发布正文和话题标签。
- `export_cards.py`：按实际展示页数导出 PNG、总览图，并检查文字越界、图片缺失、正文与页脚重叠。
- `requirements.txt`：导出所需的 Python 依赖。
- `preview.jpg`：本版七图总览。
- `SOURCES.md`：论文出处、图卡与章节对应关系。
- `checks/layout-report.json`：本次导出的排版检查记录。

## 修改文字与样式

主要预览为渲染后的 `cards.html`，图卡纵向排列并适配浏览器宽度。可直接用浏览器打开；当前本地地址为 `http://127.0.0.1:8001/cards.html`。

直达当前第 4、6、7 页分别使用 `#card-5`、`#card-7`、`#card-8`。导出尺寸仍为 1080 × 1440。

用 VS Code 等文本编辑器打开 `cards.html`。搜索 `id="card-3"` 可定位原第 3 张、现第 2 张图；编辑其中的标题和段落，保存后刷新浏览器。`<br/>` 表示手动换行，保留它即可维持当前行数。当前第 5、6 张图的结果数值、评估协议和 SOURCES.md 中的出处应一起维护。

修改 `cards.css` 可调整字号、间距和颜色。画布固定为 1080 × 1440；文件后半部分的 `#card-N` 规则是原版保留的逐页修正。

图片位于 `assets/`，替换图片文件或修改 HTML 的 `src` 即可。无需重新生成已有插图。

## 导出 PNG

要求 Python 3.10 或更新版本。在本目录执行：

```bash
python -m pip install -r requirements.txt
python -m playwright install chromium
python export_cards.py
```

默认输出到 `output/`：

```text
01.png … 07.png       1080 × 1440
preview.jpg           七图总览
layout-report.json    图片和排版检查
```

导出双倍分辨率（2160 × 2880）：

```bash
python export_cards.py --scale 2
```

只检查，不导出图片：

```bash
python export_cards.py --check-only
```

也可指定本机已安装的 Chrome，无需安装另一份浏览器。例如 macOS：

```bash
python export_cards.py --browser "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
```

脚本在内存中内嵌本地 CSS 和图片后渲染；不访问外部页面，不上传素材，也不需要本地 HTTP 服务器。导出失败时会返回非零退出码；排版问题写在 `layout-report.json` 中。

## 字体与复现

字体使用本机的 Georgia、Noto Serif CJK SC / 宋体以及 Noto Sans CJK SC / 苹方等回退字体；源码包不包含字体文件。不同电脑的字形和换行可能略有差异。已导出的 PNG 字形固定，不受接收设备影响。

## 素材与内容

本包是图文排版源码，不包含 DuetRNA 模型训练代码。第 1 张的 RNA 图为结构示意；当前第 4 张的三幅状态图为双框架插值示意。论文图、实验数据与原项目素材的出处统一见 `SOURCES.md`；图上只保留理解内容所需的说明。
