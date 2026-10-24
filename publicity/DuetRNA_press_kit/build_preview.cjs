/* Render portable equations and a local review page from the Markdown exports.
   Dependencies: playwright-core, marked; set NODE_PATH if using a bundled runtime. */
const fs = require('fs');
const path = require('path');
const {pathToFileURL} = require('url');
const {chromium} = require('playwright-core');

async function main() {
  const output = path.resolve(process.argv[2] || __dirname);
  const manifest = JSON.parse(fs.readFileSync(path.join(output, 'markdown_manifest.json'), 'utf8'));
  const {marked} = await import(pathToFileURL(require.resolve('marked')).href);
  const chrome = process.env.CHROME_PATH;
  const browser = await chromium.launch({...(chrome ? {executablePath:chrome} : {channel:'chrome'}), headless:true});
  const page = await browser.newPage({viewport:{width:2400,height:1400},deviceScaleFactor:2});
  await page.route(/^https?:/, route=>route.abort());
  await page.goto(pathToFileURL(path.resolve(output, manifest[0].source)).href,{waitUntil:'domcontentloaded'});
  await page.evaluate(()=>document.fonts.ready);
  await page.addStyleTag({content:`
    .article{max-width:none!important;width:auto!important;margin:0!important;padding:30px!important}
    .fm-card,.fm-math-scroll{max-width:none!important;width:max-content!important;overflow:visible!important;background:#fff!important;padding:0!important;border:0!important;box-shadow:none!important}
    .math-formula{max-width:none!important;width:max-content!important;display:block!important;font-size:26px!important}
    .katex-display{margin:0!important;padding:18px 28px!important;max-width:none!important;width:max-content!important;background:white!important;color:#17212b!important}
  `});
  const equations = page.locator('.fm-card .katex-display');
  if(await equations.count()!==7)throw new Error('Expected seven equations');
  const formulaSizes = [];
  for(let i=0;i<7;i++) {
    const el = equations.nth(i);
    await el.screenshot({path:path.join(output,'media',`equation_${String(i+1).padStart(2,'0')}.png`)});
    formulaSizes.push(await el.boundingBox());
  }
  await browser.close();
  const style=`
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#fff;color:#253038;font-family:Georgia,'Noto Serif SC','Songti SC',SimSun,serif;font-size:18px;line-height:1.85}
    article{max-width:900px;padding:54px 36px 90px;margin:auto}h1{font-size:38px;line-height:1.35;margin:0 0 28px}h2{font-size:28px;margin:58px 0 22px;line-height:1.4}h3{font-size:22px;margin:32px 0 16px}h4{font-size:19px;margin:32px 0 18px}p{margin:18px 0}a{color:#386880;text-decoration-thickness:1px;text-underline-offset:3px}img{display:block;max-width:100%;height:auto;margin:28px auto}img[alt^="公式"],img[alt^="Equation"]{max-height:200px;width:auto}table{width:100%;border-collapse:collapse;font-size:16px;margin:24px 0}th,td{padding:12px 14px;text-align:left;border-bottom:1px solid #dce2e5;vertical-align:top}th{background:#f7f9f9}blockquote{border-left:3px solid #688e89;background:#f5f8f5;margin:32px 0;padding:12px 24px}blockquote p{margin:8px 0}strong{color:#132e3b}li{margin:10px 0;overflow-wrap:anywhere}sup{font-size:.7em}code{font-size:.9em}nav{padding:14px 22px;border-bottom:1px solid #e5e9ea;font-family:system-ui,sans-serif;font-size:14px}nav{display:flex;flex-wrap:wrap;gap:8px 18px}nav a{margin:0}article{overflow-wrap:anywhere}@media(max-width:600px){article{padding:32px 20px 60px;font-size:16px}h1{font-size:29px}h2{font-size:24px}table{font-size:13px}td,th{padding:8px 6px}}
  `;
  const previewStyle = style.replace('max-height:200px;width:auto','max-height:240px');
  for(const r of manifest){
    const markdown=fs.readFileSync(path.join(output,r.markdown),'utf8');
    let html=await marked.parse(markdown,{gfm:true});
    html=html.replace(/<img src="media\/equation_(\d+)\.png"/g,(tag,num)=>tag+' width="'+Math.round(formulaSizes[Number(num)-1].width)+'"');
    const doc=`<!doctype html><html lang="${r.language==='zh'?'zh-CN':'en'}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DuetRNA · ${r.language==='zh'?'中文博客':'Research story'}</title><style>${previewStyle}</style></head><body><nav><a href="preview.html">Video</a><a href="blog_zh.html">中文</a><a href="blog_en.html">English</a><a href="social_posts.html">发布配文</a><a href="${r.markdown}" download>Markdown ↓</a><a href="${r.latex_markdown}" download>LaTeX Markdown ↓</a></nav><article>${html}</article></body></html>`;
    fs.writeFileSync(path.join(output,`blog_${r.language}.html`),doc);
  }
  const socialMarkdown = fs.readFileSync(path.join(output,'social_posts.md'),'utf8');
  const socialBody = await marked.parse(socialMarkdown,{gfm:true});
  const socialDoc = `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DuetRNA · 社交平台配文</title><style>${previewStyle}nav{display:flex;flex-wrap:wrap;gap:8px 18px}nav a{margin:0}article{overflow-wrap:anywhere}</style></head><body><nav><a href="preview.html?variant=sampling">Video</a><a href="blog_zh.html">中文</a><a href="blog_en.html">English</a><a href="social_posts.html">发布配文</a><a href="social_posts.md" download>Markdown ↓</a></nav><article>${socialBody}</article></body></html>`;
  fs.writeFileSync(path.join(output,'social_posts.html'),socialDoc);
  // build_demo.py owns the single-video preview. Keep it when rebuilding Markdown.
  if (path.resolve(output) !== path.resolve(__dirname)) {
    fs.copyFileSync(path.join(__dirname,'preview.html'),path.join(output,'preview.html'));
  }
  fs.mkdirSync(path.join(output,'motion','build'),{recursive:true});
  fs.writeFileSync(path.join(output,'motion','build','formula_render_report.json'),JSON.stringify(formulaSizes,null,2));
  console.log('Rendered 7 formula images and bilingual Markdown previews; kept the single-video preview.');
}
main().catch(error=>{console.error(error);process.exit(1)});
