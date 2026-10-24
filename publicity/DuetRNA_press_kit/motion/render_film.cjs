const fs=require('fs'),path=require('path'),http=require('http'),{spawn}=require('child_process');
const {chromium}=require('playwright-core');
const base=__dirname,args=process.argv.slice(2),full=args.includes('--full');
function opt(k,d){let i=args.indexOf(k);return i<0?d:args[i+1]}
const output=opt('--output',path.join(base,'..','DuetRNA_31s.mp4'));
const ffmpeg=opt('--ffmpeg',process.env.FFMPEG_PATH || 'ffmpeg');
const build=opt('--build',path.join(base,'build','v27')),times=opt('--times','0,8.8,12.65,13.22,13.6,13.9,13.95,14.1,14.25,14.4,14.47,14.5,14.5333333333,14.6,17.4,20.27,28,30.5').split(',').map(Number);
fs.mkdirSync(build,{recursive:true});
const mime={'.html':'text/html; charset=utf-8','.js':'text/javascript','.mjs':'text/javascript','.json':'application/json','.jpg':'image/jpeg','.png':'image/png'};
const server=http.createServer((req,res)=>{try{const u=new URL(req.url,'http://localhost'),p=path.resolve(base,'.'+decodeURIComponent(u.pathname));if(!p.startsWith(base+path.sep)){res.writeHead(403);return res.end()}res.setHeader('Content-Type',mime[path.extname(p)]||'application/octet-stream');res.end(fs.readFileSync(p))}catch(e){res.writeHead(404);res.end(String(e))}});
(async()=>{
 await new Promise(r=>server.listen(0,'127.0.0.1',r));
 const chrome=opt('--chrome',process.env.CHROME_PATH);
 const browser=await chromium.launch({...(chrome ? {executablePath:chrome} : {channel:'chrome'}),headless:true,args:[...(process.platform==='win32'?['--use-angle=d3d11']:[]),'--enable-gpu','--ignore-gpu-blocklist','--hide-scrollbars','--force-device-scale-factor=1']});
 const page=await browser.newPage({viewport:{width:1920,height:1080},deviceScaleFactor:1});let errors=[];page.on('pageerror',e=>{errors.push(String(e));console.error(e)});page.on('console',m=>{if(m.type()==='error')console.error(m.text())});
 await page.goto(`http://127.0.0.1:${server.address().port}/film.html?capture=1`,{waitUntil:'load'});await page.waitForFunction(()=>window.ready,null,{timeout:60000});
 const duration=await page.evaluate(()=>window.frameInfo().duration),totalFrames=Math.round(duration*30);
 const scatterAudit=await page.evaluate(()=>window.scatterAudit());
 if(scatterAudit.startError>1e-5||scatterAudit.endError>1e-5||scatterAudit.maxBoundarySpeed>.01||scatterAudit.handoffGapSeconds!==0||scatterAudit.firstFlowStepPx<=0)throw new Error(JSON.stringify(scatterAudit));
 fs.writeFileSync(path.join(build,'scatter_audit.json'),JSON.stringify(scatterAudit,null,2));
 console.log('Scatter continuity: '+JSON.stringify(scatterAudit));
 const reports=[];
 for(const t of times){await page.evaluate(s=>window.renderFrame(s),t);await page.screenshot({path:path.join(build,`frame_${t.toFixed(2).padStart(5,'0')}.jpg`),type:'jpeg',quality:96});reports.push(await page.evaluate(()=>({info:window.frameInfo(),text:[...document.querySelectorAll('.label')].filter(e=>Number(e.style.opacity)>.1).map(e=>({id:e.id,text:e.textContent,opacity:e.style.opacity,box:{x:e.getBoundingClientRect().x,y:e.getBoundingClientRect().y,w:e.getBoundingClientRect().width,h:e.getBoundingClientRect().height}}))})));console.log(`Checked ${t}s`);}
 if(errors.length)throw new Error(errors.join('\n'));
 fs.writeFileSync(path.join(build,'frame_audit.json'),JSON.stringify(reports,null,2));
 if(full){
  const log=fs.openSync(path.join(build,'encode.log'),'w');
  const encoder=spawn(ffmpeg,['-hide_banner','-loglevel','warning','-y','-f','image2pipe','-framerate','30','-vcodec','mjpeg','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','17','-vf','scale=in_range=full:out_range=limited:out_color_matrix=bt709,format=yuv420p','-color_range','tv','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709','-movflags','+faststart','-metadata','title=DuetRNA | Dual-Frame Flow Matching for RNA Co-Design',output],{stdio:['pipe','ignore',log]});
  encoder.stdin.on('error',e=>console.error(e));const closed=new Promise((r,j)=>{encoder.on('error',j);encoder.on('close',code=>code?j(new Error('FFmpeg exit '+code)):r())});
  for(let i=0;i<totalFrames;i++){await page.evaluate(s=>window.renderFrame(s),i/30);const jpg=await page.screenshot({type:'jpeg',quality:96});if(!encoder.stdin.write(jpg))await new Promise(r=>encoder.stdin.once('drain',r));if(i%60===0)console.log(`Rendered ${(i/30).toFixed(0)}/${duration} s`);}
  encoder.stdin.end();await closed;fs.closeSync(log);console.log('EXPORTED '+output);
 }
 await browser.close();server.close();
})().catch(e=>{console.error(e);server.close();process.exit(1)});
