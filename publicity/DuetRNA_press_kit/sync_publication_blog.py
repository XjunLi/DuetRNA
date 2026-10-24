#!/usr/bin/env python3
"""Sync approved Markdown prose into the existing interactive publication shell.
Requires BeautifulSoup and Node/marked; use NODE_PATH for a bundled runtime.
"""
from pathlib import Path
from copy import deepcopy
from datetime import datetime
import base64, hashlib, html, json, re, shutil, subprocess
from bs4 import BeautifulSoup, Tag
ROOT=Path(__file__).resolve().parent
PUBLICITY=ROOT.parent
BLOG=PUBLICITY/'DuetRNA_blog'
FILES={'zh':'DuetRNA_互动研究手记_v2.html','en':'DuetRNA_Interactive_Research_Notes_EN.html'}
SUBTITLE={'zh':'我们为什么给 RNA 两个坐标系？','en':'Why do we give RNA two coordinate frames?'}
IDS=['wave','representation','evidence','method','results','perspective','challenges','resources']
TOC={'zh':['联合设计的任务','碱基与核糖','表示的互补性','双系流匹配','生成结果与消融','表示设计的思考','共同挑战'],'en':['The co-design task','Bases and sugars','Complementary frames','Dual-frame flow matching','Results and ablations','Representation design','Shared challenges']}
CSS="""
.publication-table{margin:28px 0;overflow-x:auto;border:1px solid var(--line,#e0e5da);border-radius:10px}
.publication-table table{border-collapse:collapse;width:100%;font:14px/1.8 var(--sans)}
.publication-table th,.publication-table td{padding:13px 16px;text-align:left;border-bottom:1px solid var(--line,#e0e5da);vertical-align:top}
.publication-table thead{background:#eff3e9;color:var(--ink)}
.publication-table tbody tr:last-child td{border-bottom:0}
.publication-table strong{color:var(--ink)}
.publication-reference{font:14px/1.9 var(--sans);color:var(--muted)}
.publication-reference a{color:var(--ink)}
.publication-reflection{margin:30px 0;padding:25px 29px;background:linear-gradient(120deg,#eff3e9,#f8f5ee);border-left:3px solid #94a68d;border-radius:0 14px 14px 0}
.section .publication-reflection p{margin:0}
#challenges{padding-bottom:60px}
@media(max-width:600px){.publication-table th,.publication-table td{padding:10px 9px;font-size:12px}.publication-reflection{padding:21px}}
"""
def norm(value):return re.sub(r'\s+','',value).replace('[','').replace(']','')
def render_markdown(path):
 code="const fs=require('fs');const {pathToFileURL}=require('url');(async()=>{const {marked}=await import(pathToFileURL(require.resolve('marked')).href);process.stdout.write(await marked.parse(fs.readFileSync(process.argv[1],'utf8'),{gfm:true}));})().catch(e=>{console.error(e);process.exit(1)});"
 return subprocess.run([shutil.which('node') or r'C:\Program Files\nodejs\node.exe','-e',code,str(path)],check=True,capture_output=True,encoding='utf-8').stdout

def transform_refs(node,lang):
 for a in list(node.select('a')):
  match=re.fullmatch(r'\[(\d+)\]',a.get_text())
  if match:
   n=match[1];a['href']='#ref-'+n;a['aria-label']=('参考资料 ' if lang=='zh' else 'Reference ')+n;a.string=n
   a.wrap(BeautifulSoup('<sup class="refs"></sup>','html.parser').sup)

def main():
 backup=PUBLICITY/'archive'/('before-blog-sync-'+datetime.now().strftime('%Y%m%d-%H%M%S'));backup.mkdir(parents=True)
 audit=ROOT/'audits'/'publication-20261022';audit.mkdir(parents=True,exist_ok=True)
 manifest_path=ROOT/'markdown_manifest.json';manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
 all_reports=[]
 for row in manifest:
  lang=row['language'];md_path=ROOT/row['markdown'];tex_path=ROOT/row['latex_markdown'];page_path=ROOT/f'blog_{lang}.html';published_path=BLOG/FILES[lang]
  for path in [md_path,tex_path,page_path,published_path]:shutil.copy2(path,backup/path.name)
  for path in [md_path,tex_path]:
   text=path.read_text(encoding='utf-8')
   text,count=re.subn(r'^\*—.*\*$',lambda m:'*'+('——' if lang=='zh' else '— ')+SUBTITLE[lang]+'*',text,count=1,flags=re.M)
   assert count==1,path
   path.write_text(text,encoding='utf-8',newline='\n')
  rendered=render_markdown(md_path)
  simple=page_path.read_text(encoding='utf-8');widths=dict(re.findall(r'equation_(\d+)\.png" width="(\d+)"',simple))
  portable=re.sub(r'<img src="media/equation_(\d+)\.png"',lambda m:m[0]+(' width="'+widths[m[1]]+'"' if m[1] in widths else ''),rendered)
  simple,count=re.subn(r'(?<=<article>).*?(?=</article>)',lambda m:portable,simple,count=1,flags=re.S);assert count==1
  page_path.write_text(simple,encoding='utf-8',newline='\n')
  raw=published_path.read_text(encoding='utf-8');old=BeautifulSoup(raw,'html.parser')
  assert old.select_one('#resources').find_parent('main') is not None
  equations=[deepcopy(x) for x in old.select('.fm-card .katex-display')];assert len(equations)==7
  widget=deepcopy(old.select_one('#flow-demo'));widget['data-publication-visual']='flow-demo'
  document=BeautifulSoup(rendered,'html.parser');children=[x for x in document.contents if isinstance(x,Tag)];groups=[]
  for node in children:
   if node.name=='h2':groups.append([node])
   elif groups:groups[-1].append(node)
  assert len(groups)==len(IDS)
  sections=[];block_report=[];figure_count=0;equation_count=0
  for section_i,(section_id,nodes) in enumerate(zip(IDS,groups)):
   section=BeautifulSoup('<section></section>','html.parser').section;section['id']=section_id;section['class']='resources' if section_id=='resources' else 'section'
   if section_id!='resources':section.append(BeautifulSoup(f'<div class="section-head"><span class="section-num">{section_i+1:02d}</span></div>','html.parser').div)
   consumed=set()
   for index,node in enumerate(nodes):
    if index in consumed:continue
    original=deepcopy(node);element=deepcopy(node);block_id=f'{section_id}-{index}';element['data-md-block']=block_id;image=element.find('img')
    if image:
     src=image['src'];eq=re.fullmatch(r'media/equation_(\d+)\.png',src)
     if eq:
      n=int(eq[1])-1
      element=BeautifulSoup('<div class="fm-card"><div class="fm-math-scroll" tabindex="0"><span class="math-formula"></span></div></div>','html.parser').div
      element['data-md-block']=block_id;element['data-equation-index']=str(n+1);element.select_one('.math-formula').append(deepcopy(equations[n]));equation_count+=1
     else:
      assert re.fullmatch(r'media/blog_figure_0[1-3]\.png',src),src
      assert index+1<len(nodes) and nodes[index+1].name=='p'
      caption=deepcopy(nodes[index+1]);caption.name='figcaption';caption['data-md-block']=f'{section_id}-{index+1}';transform_refs(caption,lang);consumed.add(index+1)
      element=BeautifulSoup('<figure class="figure wide"><button class="zoomable" type="button"></button></figure>','html.parser').figure;element['data-md-block']=block_id
      button=element.button;button['aria-label']='放大图片' if lang=='zh' else 'Enlarge image'
      image=deepcopy(image);image['src']='data:image/png;base64,'+base64.b64encode((ROOT/src).read_bytes()).decode();image['loading']='lazy'
      button.append(image);element.append(caption);figure_count+=1
      block_report.append({'id':f'{section_id}-{index+1}','matches':norm(nodes[index+1].get_text())==norm(caption.get_text())})
    elif element.name=='table':
     container=BeautifulSoup('<div class="publication-table"></div>','html.parser').div;container.append(element);element=container
    elif element.name=='ol' and section_id=='resources':
     for n,li in enumerate(element.find_all('li',recursive=False),1):li['id']=f'ref-{n}';li['class']='publication-reference'
    elif section_id=='perspective' and index==len(nodes)-1:
     wrapper=BeautifulSoup('<aside id="reflection" class="publication-reflection"></aside>','html.parser').aside;wrapper.append(element);element=wrapper
    transform_refs(element,lang)
    if not image:
     assert norm(original.get_text())==norm(element.get_text()),block_id
     block_report.append({'id':block_id,'matches':True})
    section.append(element)
    if section_id=='method' and index==1:section.append(deepcopy(widget))
   sections.append(str(section))
  assert equation_count==7 and figure_count==3
  replacement='<main class="article" id="article">'+'\n'.join(sections)+'</main>'
  raw,count=re.subn(r'<main\b[^>]*id="article"[^>]*>.*?</main>',lambda m:replacement,raw,count=1,flags=re.S);assert count==1
  raw,count=re.subn(r'(<p class="hero-deck">).*?(</p>)',lambda m:m[1]+html.escape(SUBTITLE[lang])+m[2],raw,count=1,flags=re.S);assert count==1
  toc='<aside class="toc" aria-label="'+('文章目录' if lang=='zh' else 'Contents')+'">'+''.join(f'<a href="#{sid}"><span>{i+1:02d}</span>{TOC[lang][i]}</a>' for i,sid in enumerate(IDS[:-1]))+'</aside>'
  raw,count=re.subn(r'<aside\b(?=[^>]*class="toc")[^>]*>.*?</aside>',lambda m:toc,raw,count=1,flags=re.S);assert count==1
  raw=re.sub(r'<style id="publication-markdown-style">.*?</style>','',raw,flags=re.S)
  raw=raw.replace('</head>','<style id="publication-markdown-style">'+CSS+'</style></head>')
  published_path.write_text(raw,encoding='utf-8',newline='\n')
  row['editorial_revision']='user-final-subtitle-20261022';row['source_sha256']=hashlib.sha256(raw.encode()).hexdigest()
  for key,path in [('markdown_sha256',md_path),('latex_markdown_sha256',tex_path)]:row[key]=hashlib.sha256(path.read_text(encoding='utf-8').encode()).hexdigest()
  all_reports.append({'language':lang,'body_blocks':len(block_report),'all_body_blocks_match':all(x['matches'] for x in block_report),'equations':equation_count,'figures':figure_count,'subtitle':SUBTITLE[lang],'blocks':block_report})
 shutil.copy2(manifest_path,backup/manifest_path.name)
 manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
 (audit/'source-sync.json').write_text(json.dumps({'backup':str(backup),'reports':all_reports},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps([{k:v for k,v in r.items() if k!='blocks'} for r in all_reports],ensure_ascii=False,indent=2))
if __name__=='__main__':main()
