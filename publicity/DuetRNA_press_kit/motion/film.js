import * as THREE from 'three';
import {makeScatterMotion,startupProgress} from './scatter_motion.mjs';

// A deterministic film with the near-white tail trimmed. Only the recorded sampling states are used.
const DATA=await(await fetch('./data/film_data.json')).json();
const W=1920,H=1080,DURATION=922/30, RED='#d72c40',BLUE='#204fe0',INK='#24473d';
const stage=document.querySelector('#stage'),labels=document.querySelector('#labels'),svg=document.querySelector('#graphics');
const cfg=new URLSearchParams(location.search);document.body.classList.toggle('capture',cfg.has('capture'));
function fit(){const s=Math.min(innerWidth/W,innerHeight/H);stage.style.transform=`translate(${(innerWidth-W*s)/2}px,${(innerHeight-H*s)/2}px) scale(${s})`;}
addEventListener('resize',fit);fit();
const renderer=new THREE.WebGLRenderer({canvas:document.querySelector('#gl'),antialias:true,alpha:true,preserveDrawingBuffer:true});
renderer.setSize(W,H,false);renderer.setPixelRatio(1);renderer.setClearColor(0x000000,0);renderer.outputColorSpace=THREE.SRGBColorSpace;
renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.18;
const scene=new THREE.Scene(),camera=new THREE.OrthographicCamera(-W/2,W/2,H/2,-H/2,1,10000);camera.position.set(0,0,5000);camera.lookAt(0,0,0);
scene.add(new THREE.AmbientLight(0xffffff,1.55));
for(const [color,intensity,x,y,z] of [[0xfff9ee,2.6,-500,800,1400],[0xc5daff,1.1,800,100,700],[0xffffff,1.6,0,-400,-900]]){const l=new THREE.DirectionalLight(color,intensity);l.position.set(x,y,z);scene.add(l);}
const V=()=>new THREE.Vector3(),Q=()=>new THREE.Quaternion(),Y=new THREE.Vector3(0,1,0),one=new THREE.Vector3(1,1,1),dummy=new THREE.Object3D();
const clamp=v=>Math.max(0,Math.min(1,v)),smooth=v=>{v=clamp(v);return v*v*v*(v*(v*6-15)+10)},ramp=(t,a,b)=>smooth((t-a)/(b-a));
const lerp=(a,b,t)=>a+(b-a)*t, vlerp=(a,b,t)=>a.clone().lerp(b,t);
const palette=['#D4854A','#D06B60','#3A9E9C','#8B72B0','#5A8DB8'].map(c=>new THREE.Color(c));
const grey=new THREE.Color('#c5c9c2'),bondColor=new THREE.Color('#76817e');
const sphereGeo=new THREE.SphereGeometry(1,14,10),cylGeo=new THREE.CylinderGeometry(1,1,1,9),coneGeo=new THREE.ConeGeometry(1,1,12);
function material(color,flat=false){const m=flat?new THREE.MeshBasicMaterial({color,transparent:true,toneMapped:false,depthTest:false,depthWrite:false}):new THREE.MeshPhongMaterial({color,shininess:48,specular:0x6d8282,transparent:true});m.onBeforeCompile=shader=>{
 shader.vertexShader='attribute float instanceOpacity; varying float vInstanceOpacity;\n'+shader.vertexShader;
 shader.vertexShader=shader.vertexShader.replace('#include <begin_vertex>','#include <begin_vertex>\nvInstanceOpacity=instanceOpacity;');
 shader.fragmentShader='varying float vInstanceOpacity;\n'+shader.fragmentShader;
 shader.fragmentShader=shader.fragmentShader.replace('#include <color_fragment>','#include <color_fragment>\ndiffuseColor.a*=vInstanceOpacity; if(diffuseColor.a<0.008) discard;');
};return m;}
function instances(geo,n,color=0xffffff,flat=false){const g=geo.clone();g.setAttribute('instanceOpacity',new THREE.InstancedBufferAttribute(new Float32Array(n).fill(1),1));const m=new THREE.InstancedMesh(g,material(color,flat),n);m.frustumCulled=false;m.renderOrder=flat?10:0;m.instanceMatrix.setUsage(THREE.DynamicDrawUsage);for(let i=0;i<n;i++)m.setColorAt(i,new THREE.Color(color));m.userData.alpha=g.getAttribute('instanceOpacity');return m;}
function put(mesh,i,p,q,s,alpha,color){dummy.position.copy(p);dummy.quaternion.copy(q);dummy.scale.copy(s);dummy.updateMatrix();mesh.setMatrixAt(i,dummy.matrix);mesh.userData.alpha.array[i]=alpha;if(color)mesh.setColorAt(i,color);}
function flush(mesh){mesh.instanceMatrix.needsUpdate=true;mesh.userData.alpha.needsUpdate=true;if(mesh.instanceColor)mesh.instanceColor.needsUpdate=true;}
const identity=Q(),origin=V(),red=new THREE.Color(RED),blue=new THREE.Color(BLUE);

class Molecule{
 constructor({heroOnly=false}={}){
  this.root=new THREE.Group();this.content=new THREE.Group();this.root.add(this.content);scene.add(this.root);
  this.ids=DATA.atoms.map((a,i)=>i).filter(i=>!heroOnly||DATA.atoms[i][0]===DATA.hero);
  const ids=new Set(this.ids);this.bondIds=DATA.bonds.filter(([a,b])=>ids.has(a)&&ids.has(b));
  this.resIds=heroOnly?[DATA.hero]:Array.from({length:DATA.length},(_,i)=>i);
  this.atoms=instances(sphereGeo,this.ids.length);this.bonds=instances(cylGeo,this.bondIds.length);this.shafts=instances(cylGeo,this.resIds.length*6,0xffffff,true);this.tips=instances(coneGeo,this.resIds.length*6,0xffffff,true);
  this.content.add(this.atoms,this.bonds,this.shafts,this.tips);
  this.positions=DATA.atoms.map(()=>V());this.poseP=Array.from({length:DATA.length},()=>[V(),V()]);this.poseQ=Array.from({length:DATA.length},()=>[Q(),Q()]);this.last=-2;this.lastAxes='';
 }
 setPose(t,spread=0,scatter=0){
  if(this.last===t&&this.lastSpread===spread&&this.lastScatter===scatter)return;this.last=t;this.lastSpread=spread;this.lastScatter=scatter;const z=clamp(t)*(DATA.poses.length-1),lo=Math.floor(z),hi=Math.min(lo+1,DATA.poses.length-1),w=z-lo;
  for(const i of this.resIds)for(let j=0;j<2;j++){const a=DATA.poses[lo][i][j],b=DATA.poses[hi][i][j];this.poseP[i][j].set(a[0],a[1],a[2]).lerp(new THREE.Vector3(b[0],b[1],b[2]),w);this.poseQ[i][j].set(...a.slice(3)).slerp(new THREE.Quaternion(...b.slice(3)),w);}
  // Only the representation transition expands the paired frames. Recorded flow states stay untouched.
  if(spread)for(const i of this.resIds){const offset=this.poseP[i][0].clone().add(this.poseP[i][1]).multiplyScalar(.5).sub(new THREE.Vector3(...DATA.centre)).multiplyScalar(spread);this.poseP[i][0].add(offset);this.poseP[i][1].add(offset);}
  // Display-only noise initialization before the forward sampling sequence.
  // It finishes at the exact first recorded frame, without altering DATA.poses.
  if(scatter)for(const i of this.resIds)for(let j=0;j<2;j++){
   const p=DATA.poses[0][i][j];this.poseP[i][j].lerp(new THREE.Vector3(...p.slice(0,3)),scatter);this.poseQ[i][j].slerp(new THREE.Quaternion(...p.slice(3)),scatter);
  }
  for(const id of this.ids){const a=DATA.atoms[id],j=a[1]===0?1:0;this.positions[id].set(a[2],a[3],a[4]).applyQuaternion(this.poseQ[a[0]][j]).add(this.poseP[a[0]][j]);}
 }
 place(x,y,s,q,center){this.root.position.set(x-W/2,H/2-y,0);this.root.scale.setScalar(s);this.root.quaternion.copy(q);this.content.position.copy(center).negate();}
 style({atomAlpha=1,frameAlpha=0,heroFocus=0,baseLight=1,sugarFrame=1,baseFrame=1,allFrames=false,otherFrameAlpha=1,axisLength=2.5,axisRadius=.055,atomRadius=1,frameResidue=DATA.hero,frameScales=null}={}){
  this.atoms.visible=this.bonds.visible=atomAlpha>0.003;this.shafts.visible=this.tips.visible=frameAlpha>0.003;
  if(this.atoms.visible){
   for(let k=0;k<this.ids.length;k++){
    const id=this.ids[k],a=DATA.atoms[id];let c=palette[a[5]].clone(),aAlpha=atomAlpha;
    if(a[0]!==DATA.hero){c.lerp(grey,heroFocus*.8);aAlpha*=1-heroFocus;}
    else if(a[1]===1)c.lerp(grey,1-baseLight);
    const radius=(a[6]==='P'?.39:a[5]===1||a[5]===4?.31:.29)*atomRadius;
    put(this.atoms,k,this.positions[id],identity,new THREE.Vector3(radius,radius,radius),aAlpha,c);
   }
   for(let k=0;k<this.bondIds.length;k++){
    const [ia,ib]=this.bondIds[k],a=DATA.atoms[ia],b=DATA.atoms[ib],pa=this.positions[ia],pb=this.positions[ib],d=pb.clone().sub(pa),len=d.length();let alpha=atomAlpha;
    if(a[0]!==DATA.hero||b[0]!==DATA.hero)alpha*=1-heroFocus;
    // Never draw an artificial long covalent bond between noisy rigid fragments.
    if((a[0]!==b[0]||a[1]!==b[1])&&len>2.4)alpha=0;
    const color=bondColor.clone();if(a[1]===1&&b[1]===1)color.lerp(grey,1-baseLight);
    put(this.bonds,k,pa.clone().add(pb).multiplyScalar(.5),Q().setFromUnitVectors(Y,d.normalize()),new THREE.Vector3(.095*atomRadius,len,.095*atomRadius),alpha,color);
   }
   flush(this.atoms);flush(this.bonds);
  }
  if(this.shafts.visible){
   let k=0;
   for(const i of this.resIds)for(let j=0;j<2;j++)for(let axis=0;axis<3;axis++,k++){
    const p=this.poseP[i][j],rot=this.poseQ[i][j];const d=new THREE.Vector3(axis===0?1:0,axis===1?1:0,axis===2?1:0).applyQuaternion(rot);
    const q=Q().setFromUnitVectors(Y,d),col=j===0?blue:red;
    let a=frameAlpha*(j===0?baseFrame:sugarFrame);if(i!==frameResidue)a*=allFrames?otherFrameAlpha:0;
    const glyphScale=frameScales?.[i]?.[j]??1,shaft=axisLength*glyphScale*.84,head=axisLength*glyphScale*.16,radius=axisRadius*glyphScale;
    put(this.shafts,k,p.clone().addScaledVector(d,shaft/2),q,new THREE.Vector3(radius,shaft,radius),a,col);
    put(this.tips,k,p.clone().addScaledVector(d,shaft+head/2),q,new THREE.Vector3(radius*2.6,head,radius*2.6),a,col);
   }
   flush(this.shafts);flush(this.tips);
  }
 }
 project(p){this.content.updateWorldMatrix(true,false);return p.clone().applyMatrix4(this.content.matrixWorld).project(camera).multiply(new THREE.Vector3(W/2,-H/2,1)).add(new THREE.Vector3(W/2,H/2,0));}
}

const main=new Molecule(),moving=new Molecule(),scatter=new Molecule();
const checkpoints=[0,1/3,2/3,1],xs=[240,720,1200,1680],still=checkpoints.slice(0,3).map(()=>new Molecule());
const center=new THREE.Vector3(...DATA.centre),heroCenter=new THREE.Vector3(...DATA.hero_centre),canon=new THREE.Quaternion(...DATA.q_canon);
const flowQ=Q().setFromEuler(new THREE.Euler(-.05,.1,-.055));
const initialQ=Q().setFromEuler(new THREE.Euler(.12,-.48,-.1));
const flowViews=DATA.poses.map(row=>{
 const points=row.flatMap(r=>r.map(p=>new THREE.Vector3(...p.slice(0,3))));
 const c=points.reduce((s,p)=>s.add(p),V()).multiplyScalar(1/points.length);
 const box=new THREE.Box3().setFromPoints(points.map(p=>p.clone().sub(c).applyQuaternion(flowQ)));
 const size=box.getSize(V());c.add(box.getCenter(V()).applyQuaternion(flowQ.clone().invert()));return {center:c,scale:Math.min(6.5,360/(size.x+10),360/(size.y+10))};
});
function flowView(p){const k=p*(flowViews.length-1),a=Math.floor(k),b=Math.min(a+1,flowViews.length-1),f=k-a;return {center:vlerp(flowViews[a].center,flowViews[b].center,f),scale:lerp(flowViews[a].scale,flowViews[b].scale,f)};}
// The introductory dispersion uses a fixed display-space camera. Each frame
// follows a separate smooth path into the exact first recorded sampling pose.
const scatterMotion=makeScatterMotion(DATA.poses,flowView(1),flowView(0),flowQ);
const scatterScales=Array.from({length:DATA.length},()=>[1,1]);
function updateScatter(time){
 for(let i=0;i<DATA.length;i++)for(let j=0;j<2;j++)scatterScales[i][j]=scatterMotion.sample(i,j,time,scatter.poseP[i][j],scatter.poseQ[i][j]);
 scatter.place(xs[0],426,1,identity,origin);
 scatter.style({atomAlpha:0,frameAlpha:1,allFrames:true,axisLength:scatterMotion.axisLength,axisRadius:scatterMotion.axisRadius,frameScales:scatterScales});
}
const blurCanvas=document.createElement('canvas');blurCanvas.width=W;blurCanvas.height=H;
Object.assign(blurCanvas.style,{position:'absolute',inset:'0',width:W+'px',height:H+'px',zIndex:'2',pointerEvents:'none',opacity:'0'});
renderer.domElement.after(blurCanvas);
const blurContext=blurCanvas.getContext('2d',{alpha:true});
const SHUTTER_SECONDS=1/60,BLUR_SAMPLES=8;
function drawScatterExposure(time){
 blurContext.clearRect(0,0,W,H);blurContext.globalCompositeOperation='lighter';blurContext.globalAlpha=1/BLUR_SAMPLES;
 for(let k=0;k<BLUR_SAMPLES;k++){
  updateScatter(time+((k+.5)/BLUR_SAMPLES-.5)*SHUTTER_SECONDS);
  renderer.render(scene,camera);blurContext.drawImage(renderer.domElement,0,0);
 }
 blurContext.globalAlpha=1;blurContext.globalCompositeOperation='source-over';
 updateScatter(time);renderer.domElement.style.visibility='hidden';blurCanvas.style.opacity='1';
}

// A continuous cartoon trace provides a visual bridge to the existing sample gallery.
const gradient=['#ff4237','#ff9c36','#ffe84b','#b1e84a','#66c874','#83cfd1'].map(c=>new THREE.Color(c));
function chainColor(u){const z=clamp(u)*(gradient.length-1),i=Math.min(Math.floor(z),gradient.length-2);return gradient[i].clone().lerp(gradient[i+1],z-i);}
const finalWorld=DATA.atoms.map(a=>{const p=DATA.poses.at(-1)[a[0]][a[1]===0?1:0];return new THREE.Vector3(a[2],a[3],a[4]).applyQuaternion(new THREE.Quaternion(...p.slice(3))).add(new THREE.Vector3(...p.slice(0,3)))});
const c4=DATA.residue_types.map((_,i)=>finalWorld[DATA.atoms.findIndex(a=>a[0]===i&&a[6]==="C4'")]);
const cartoon=new THREE.Group();moving.content.add(cartoon);cartoon.visible=false;
const curve=new THREE.CatmullRomCurve3(c4,false,'centripetal'),segments=DATA.length*6,radial=12;
const transported=curve.computeFrenetFrames(segments,false),ribbonPositions=[],ribbonColors=[],ribbonIndices=[];
// An elliptical ribbon section, rather than the old round tube, bridges to the original cartoons.
for(let i=0;i<=segments;i++){
 const c=curve.getPointAt(i/segments),color=chainColor(curve.getUtoTmapping(i/segments));
 for(let j=0;j<=radial;j++){
  const a=j/radial*Math.PI*2,p=c.clone().addScaledVector(transported.normals[i],Math.cos(a)*1.30).addScaledVector(transported.binormals[i],Math.sin(a)*.20);
  ribbonPositions.push(...p.toArray());ribbonColors.push(color.r,color.g,color.b);
  if(i<segments&&j<radial){const k=i*(radial+1)+j;ribbonIndices.push(k,k+radial+1,k+1,k+1,k+radial+1,k+radial+2);}
 }
}
const tubeGeometry=new THREE.BufferGeometry();tubeGeometry.setAttribute('position',new THREE.Float32BufferAttribute(ribbonPositions,3));tubeGeometry.setAttribute('color',new THREE.Float32BufferAttribute(ribbonColors,3));tubeGeometry.setIndex(ribbonIndices);tubeGeometry.computeVertexNormals();
const cartoonMat=new THREE.MeshPhongMaterial({vertexColors:true,shininess:30,specular:0x333333,toneMapped:false,transparent:true});cartoon.add(new THREE.Mesh(tubeGeometry,cartoonMat));
const vertices=[],plateColors=[];
for(let i=0;i<DATA.length;i++){
 const lookup={};DATA.atoms.forEach((a,k)=>{if(a[0]===i)lookup[a[6]]=finalWorld[k]});
 for(const names of [['N1','C2','N3','C4','C5','C6'],...(['A','G'].includes(DATA.residue_types[i])?[['C4','C5','N7','C8','N9']]:[])]){
  if(names.some(n=>!lookup[n]))continue;const pts=names.map(n=>lookup[n]),c=pts.reduce((s,p)=>s.add(p),V()).multiplyScalar(1/pts.length),color=chainColor(i/(DATA.length-1));
  for(let j=0;j<pts.length;j++){for(const p of [c,pts[j],pts[(j+1)%pts.length]]){vertices.push(...p.toArray());plateColors.push(color.r,color.g,color.b);}}
 }
}
const plates=new THREE.BufferGeometry();plates.setAttribute('position',new THREE.Float32BufferAttribute(vertices,3));plates.setAttribute('color',new THREE.Float32BufferAttribute(plateColors,3));plates.computeVertexNormals();
const plateMat=cartoonMat.clone();plateMat.side=THREE.DoubleSide;cartoon.add(new THREE.Mesh(plates,plateMat));
// Slender base connectors preserve the familiar ribbon-and-base cartoon vocabulary.
const stemPositions=[],stemColors=[],stemIndices=[];
for(let i=0;i<DATA.length;i++){
 const ids=DATA.atoms.map((a,k)=>a[0]===i&&a[1]===1?k:-1).filter(k=>k>=0);
 const end=ids.reduce((p,k)=>p.add(finalWorld[k]),V()).multiplyScalar(1/ids.length),start=c4[i],d=end.clone().sub(start),length=d.length();
 const geo=new THREE.CylinderGeometry(.11,.11,length,6),q=Q().setFromUnitVectors(Y,d.normalize());geo.applyQuaternion(q);geo.translate(...start.clone().add(end).multiplyScalar(.5).toArray());
 const offset=stemPositions.length/3,color=chainColor(i/(DATA.length-1));
 stemPositions.push(...geo.attributes.position.array);for(let k=0;k<geo.attributes.position.count;k++)stemColors.push(color.r,color.g,color.b);
 stemIndices.push(...Array.from(geo.index.array,v=>v+offset));geo.dispose();
}
const stemGeometry=new THREE.BufferGeometry();stemGeometry.setAttribute('position',new THREE.Float32BufferAttribute(stemPositions,3));stemGeometry.setAttribute('color',new THREE.Float32BufferAttribute(stemColors,3));stemGeometry.setIndex(stemIndices);stemGeometry.computeVertexNormals();cartoon.add(new THREE.Mesh(stemGeometry,cartoonMat));
const frameState={};
function label(id,html,klass,style={}){const e=document.createElement('div');e.id=id;e.className='label '+klass;e.innerHTML=html;Object.assign(e.style,style);labels.appendChild(e);return e;}
const ui={
 brand:label('brand','Duet<span class="coral">RNA</span>','brand'),
 open:label('open','Duet<span class="coral">RNA</span>','opening'),
 paper:label('paper','Base-and-Sugar Dual-Frame Flow Matching<br><em>for RNA Co-Design</em>','paper-title'),
 rep:label('rep','Dual-frame representation','heading'),
 flow:label('flow','Flow matching with a relative-pose constraint','heading'),
 samples:label('samples','RNA samples generated by DuetRNA','heading'),
 sugar:label('sugar','Sugar frame','frame-label sugar',{left:'660px'}),
 base:label('base','Base frame','frame-label base',{left:'1260px'}),
 plus:label('plus','+','frame-label muted',{left:'960px',fontSize:'37px'}),
 sugarDesc:label('sugarDesc','Sugar–phosphate backbone<br>reconstruction','frame-desc',{left:'660px'}),
 baseDesc:label('baseDesc','Base pairing & stacking','frame-desc',{left:'1260px'}),
 caption:label('caption','','caption'),
 meta:label('meta','110 nt · (SE(3) × SE(3))<sup style="font-size:17px">N</sup>','meta'),
 legend:label('legend',`<i style="background:${RED}"></i>Sugar frame<i style="background:${BLUE}"></i>Base frame`,'legend'),
 con:label('con','Relative-pose supervision','constraint-title'),
 conSub:label('conSub','Rotation + translation','constraint-sub'),
 conSmall:label('conSmall','Between the two predicted endpoint frames','constraint-small'),
 conDiagram:label('conDiagram','<img src="./assets/relative_pose_paper.png" alt="Sugar and base frames with relative rotation and translation arrows">','constraint-diagram'),
};
const ticks=checkpoints.map((v,i)=>({t:label('tick'+i,`t = ${v===0?'0':v===1?'1':v.toFixed(2)}`,'tick',{left:xs[i]+'px'}),n:label('tickName'+i,i===0?'Noise':i===3?'Generated RNA':'','tick-sub',{left:xs[i]+'px'})}));
function show(e,a=1){e.style.opacity=clamp(a);}
const setText=(e,text)=>{if(e.textContent!==text)e.textContent=text};
function svgLine(a,b,color,opacity=1,width=1.5,dash=''){return `<line x1="${a.x.toFixed(2)}" y1="${a.y.toFixed(2)}" x2="${b.x.toFixed(2)}" y2="${b.y.toFixed(2)}" stroke="${color}" stroke-opacity="${opacity}" stroke-width="${width}" ${dash?`stroke-dasharray="${dash}"`:''}/>`;}
function svgPath(d,color,opacity=1,width=1.5,dash=''){return `<path d="${d}" fill="none" stroke="${color}" stroke-opacity="${opacity}" stroke-width="${width}" stroke-linecap="round" ${dash?`stroke-dasharray="${dash}"`:''}/>`;}

// The two teaching frames are projected from the actual 3D poses and drawn above
// the atoms. Screen-space arrowheads keep all six directions crisp and legible.
function projectedFrame(m,residue,j,alpha,{length=3,width=6.5,head=20}={}){
 if(alpha<.003)return '';
 const p=m.poseP[residue][j],q=m.poseQ[residue][j],o=m.project(p),color=j===0?BLUE:RED;
 let shapes='';
 for(let axis=0;axis<3;axis++){
  const d=new THREE.Vector3(axis===0?1:0,axis===1?1:0,axis===2?1:0).applyQuaternion(q),end=m.project(p.clone().addScaledVector(d,length));
  const dx=end.x-o.x,dy=end.y-o.y,len=Math.hypot(dx,dy),ux=dx/len,uy=dy/len,h=Math.min(head,len*.25),half=h*.40;
  const neck={x:end.x-ux*h*.85,y:end.y-uy*h*.85};
  const points=`${end.x},${end.y} ${end.x-ux*h-uy*half},${end.y-uy*h+ux*half} ${end.x-ux*h+uy*half},${end.y-uy*h-ux*half}`;
  shapes+=svgLine(o,neck,'#fbfaf5',1,width+4)+svgLine(o,neck,color,1,width);
  shapes+=`<polygon points="${points}" fill="${color}" stroke="#fbfaf5" stroke-width="1.4" stroke-linejoin="round"/>`;
 }
 shapes+=`<circle cx="${o.x}" cy="${o.y}" r="${width*.62}" fill="${color}" stroke="#fbfaf5" stroke-width="1.5"/>`;
 return `<g opacity="${alpha}">${shapes}</g>`;
}
function frameLeader(p,end,color,alpha){
 const d={x:end.x-p.x,y:end.y-p.y},length=Math.hypot(d.x,d.y),start={x:p.x+d.x/length*12,y:p.y+d.y/length*12};
 return `<g opacity="${alpha}">${svgLine(start,end,'#fbfaf5',1,4.5)}${svgLine(start,end,color,1,1.8)}</g>`;
}
const gallery=document.querySelector('#gallery'),galleryItems=[];
for(let i=0;i<6;i++){
 const nt=['40','70','90','110','120','140'][i],x=340+(i%3)*620,y=380+Math.floor(i/3)*434;
 const img=new Image();img.className='gallery-img';img.style.left=(x-204)+'px';img.style.top=(y-204)+'px';img.src=`./build/gallery_original/${nt}/f_0000.jpg`;gallery.appendChild(img);
 const lab=document.createElement('div');lab.className='gallery-label';lab.style.left=x+'px';lab.style.top=(y-220)+'px';lab.textContent=nt+' nt';gallery.appendChild(lab);
 galleryItems.push({img,lab,nt,last:0});
}
gallery.style.opacity=0;

window.renderFrame=async function(seconds){
 const time=Math.max(0,Math.min(DURATION,seconds)),t=time>14?time-1:time;let lines='';
 for(const e of Object.values(ui))show(e,0);ticks.forEach(a=>{show(a.t,0);show(a.n,0)});
 for(const m of [main,moving,scatter,...still])m.root.visible=false;
 renderer.domElement.style.visibility='visible';blurCanvas.style.opacity='0';
 ui.caption.style.top='994px';
 gallery.style.opacity=0;const inRep=time<14.5, inFlow=time>=12.15&&time<25.7, inGallery=time>=25&&time<=DURATION;
 if(inRep){
  main.root.visible=true;
  const zoom=ramp(time,1.7,4.5),ret=ramp(time,10.8,12.15),focus=ramp(time,2.1,3.5)*(1-ret),base=ramp(time,7.6,8.7);
  main.setPose(1);
  const rotation=Q().setFromEuler(new THREE.Euler(.12,lerp(-.48,.08,clamp(time/1.8)),-.1));
  const q=rotation.slerp(canon,zoom).slerp(flowQ,ret);
  const target=vlerp(center,heroCenter,zoom).lerp(flowView(1).center,ret);
  // Overlap the pullback and lateral travel; both decelerate with continuous velocity and acceleration.
  const slide=ramp(time,11.25,12.65),closeScale=lerp(Math.log(7),Math.log(96),zoom),sc=Math.exp(lerp(lerp(closeScale,Math.log(5.6),ret),Math.log(flowView(1).scale),slide)),px=lerp(lerp(1370,960,zoom),xs[0],slide),py=lerp(lerp(545,475,zoom),426,slide);
  main.place(px,py,sc,q,target);
  const sugar=ramp(time,4.1,4.8),out=1,all=ramp(time,12.95,13.2),closeFrame=sugar*(1-ret);
  main.style({atomAlpha:(1-ramp(time,13.0,13.22))*out,heroFocus:focus,baseLight:lerp(1,base,zoom),frameAlpha:all*out,allFrames:true,axisLength:scatterMotion.axisLength/sc,axisRadius:scatterMotion.axisRadius/sc,atomRadius:lerp(1.45,1.04,zoom)});
  if(time>=scatterMotion.start){
   main.root.visible=false;scatter.root.visible=true;updateScatter(time);
  }
  const open=1-ramp(t,1.5,2.3);show(ui.open,open);show(ui.paper,open);ui.open.style.transform=`translateY(${-18*(1-open)}px)`;ui.paper.style.transform=`translateY(${-10*(1-open)}px)`;
  const repA=ramp(time,2.2,3)*(1-ramp(time,11.75,12.12));show(ui.rep,repA);show(ui.brand,ramp(t,2.3,3.1));
  const labelA=ramp(t,4.5,5.2)*(1-ramp(t,10.5,11.2));
  show(ui.sugar,labelA);show(ui.sugarDesc,labelA);show(ui.base,labelA*base);show(ui.baseDesc,labelA*base);show(ui.plus,0);
  ui.sugar.style.left='300px';ui.sugar.style.top='700px';ui.sugarDesc.style.left='300px';ui.sugarDesc.style.top='755px';
  ui.base.style.left='1510px';ui.base.style.top='579px';ui.baseDesc.style.left='1510px';ui.baseDesc.style.top='634px';
  if(labelA>.003){
   const ps=main.project(main.poseP[DATA.hero][1]),pb=main.project(main.poseP[DATA.hero][0]);
   lines+=frameLeader(ps,{x:540,y:726},RED,labelA*.78);
   lines+=frameLeader(pb,{x:1492,y:605},BLUE,labelA*base*.78);
  }
  lines+=projectedFrame(main,DATA.hero,1,closeFrame,{length:2.6,width:7,head:21});
  lines+=projectedFrame(main,DATA.hero,0,closeFrame*base,{length:3.05,width:7,head:21});
  if(time<3){setText(ui.caption,'Introducing DuetRNA — co-designing RNA sequences and 3D structures.');show(ui.caption,1-ramp(time,2.75,3));}
  if(t>=4.6&&t<7.8){setText(ui.caption,'The sugar frame describes the sugar–phosphate backbone.');show(ui.caption,labelA);}
  if(t>=7.8&&t<10.8){setText(ui.caption,'DuetRNA adds a base frame for pairing and stacking.');show(ui.caption,labelA*base);}
 }
 if(inFlow){
  const a=ramp(time,12.15,12.85)*(1-ramp(t,24,24.7)),p=startupProgress((time-14.5)/8.6),x=lerp(xs[0],xs[3],p);
  const textA=a*(1-ramp(time,24.7,25));
  show(ui.flow,textA);show(ui.brand,a);show(ui.meta,textA);show(ui.legend,textA);
  const timelineA=textA*ramp(time,12.65,13.15);
  lines+=svgLine({x:xs[0],y:640},{x:xs[3],y:640},'#d1d7ca',timelineA,2);
  lines+=svgLine({x:xs[0],y:640},{x:x,y:640},'#738a7a',timelineA,2.4);
  for(let i=0;i<4;i++){
   const passed=time<14.5?i===0:p>=checkpoints[i]-.001,tickA=timelineA*(passed?1:.32);show(ticks[i].t,tickA);show(ticks[i].n,tickA*(i===0?ramp(time,14.2,14.5):1));
   lines+=`<circle cx="${xs[i]}" cy="640" r="${passed?4:3}" fill="${passed?'#738a7a':'#cdd3c6'}" opacity="${timelineA}"/>`;
   if(i<3&&time>=14.5&&p>=checkpoints[i]){
    const m=still[i],view=flowView(checkpoints[i]);m.root.visible=true;m.setPose(checkpoints[i]);m.place(xs[i],426,view.scale,flowQ,view.center);
    // Retain each checkpoint the instant the moving state arrives.
    m.style({atomAlpha:0,frameAlpha:.82*a,allFrames:true,axisLength:2.5,axisRadius:.14});
   }
  }
  if(time>=14.5){
   const view=flowView(p),ribbon=ramp(time,23.1,24.6),grow=ramp(time,23.65,24.65);moving.root.visible=true;moving.setPose(p);moving.place(x-48*grow,426-14*grow,view.scale*(1+.30*grow),flowQ,view.center);
   const emerge=ramp(p,.88,1),moveA=a;
   moving.style({atomAlpha:emerge*moveA*(1-ribbon),frameAlpha:(1-emerge*.78)*moveA*(1-ribbon),allFrames:true,axisLength:2.5,axisRadius:.15,atomRadius:1.35});
   cartoon.visible=ribbon>.001;cartoonMat.opacity=ribbon*moveA;plateMat.opacity=ribbon*moveA;
   lines+=`<circle cx="${x}" cy="640" r="7" fill="#f8f7f2" stroke="#4c6c59" stroke-width="2.2" opacity="${moveA}"/>`;
  }
  const ca=ramp(t,16.1,17.3)*textA;
  if(ca>.001){
   // Original paper diagram: extracted without formulas, background or outer box.
   // The fixed brace groups the trajectory; it does not track a moving frame.
   lines+=svgPath('M 240 740 C 240 747, 266 749, 296 749 L 886 749 Q 940 749, 960 765 Q 980 749, 1034 749 L 1624 749 C 1654 749, 1680 747, 1680 740','#7d9381',ca*.83,2.1,'7 7');
   show(ui.conDiagram,ca);show(ui.con,ca);show(ui.conSub,ca);show(ui.conSmall,ca);
  }
  ui.caption.style.top='1020px';
  setText(ui.caption,t<17?'Both frames flow from noise to an RNA structure.':'Relative-pose supervision couples the two predicted endpoints.');show(ui.caption,textA*ramp(time,14.5,14.8));
 }
 if(inGallery){
  const ga=ramp(t,24,24.7)*(1-ramp(time,30,31));gallery.style.opacity=ga;show(ui.samples,ga);show(ui.brand,ga);
  const n=Math.round(clamp((t-24.4)/4.6)*308);
  await Promise.all(galleryItems.map(async({img,nt,...item},i)=>{if(galleryItems[i].last!==n){img.src=`./build/gallery_original/${nt}/f_${String(n).padStart(4,'0')}.jpg`;galleryItems[i].last=n;try{await img.decode()}catch{}}}));
 }
 svg.innerHTML=lines;
 const blurred=scatter.root.visible&&time>=scatterMotion.start&&time<scatterMotion.end;
 if(blurred)drawScatterExposure(time);else renderer.render(scene,camera);
 frameState.duration=DURATION;frameState.snapshots=still.map((m,i)=>({t:checkpoints[i],visible:m.root.visible}));frameState.time=time;frameState.module=time<12.15?'representation':time<25?'constrained-flow':'generated-samples';frameState.pathType=DATA.path_type;frameState.scatterPhase=time<13.22?'forming':time<scatterMotion.start?'holding':time<(scatterMotion.start+scatterMotion.end)/2?'dispersing':time<scatterMotion.end?'decelerating':'flowing';frameState.motionBlurSamples=blurred?BLUR_SAMPLES:0;frameState.introX=time<14.5?xs[0]:null;frameState.introScale=time>=scatterMotion.start&&time<14.5?1:main.root.scale.x;frameState.transition=time<10.8?'close-up':time<11.25?'continuous camera pullback':time<12.65?'RNA moves to left start':time<13.22?'RNA becomes ordered frames':time<scatterMotion.start?'ordered frames hold':time<scatterMotion.end?'independent frames disperse directly into flow':'recorded flow or gallery';

 return frameState;
};
window.scatterAudit=()=>{
 const p=V(),q=Q(),p0=V(),p1=V(),eps=1e-4;let startError=0,endError=0,maxBoundarySpeed=0;
 for(let i=0;i<DATA.length;i++)for(let j=0;j<2;j++){
  const row=scatterMotion.rows[i][j];
  scatterMotion.sample(i,j,row.begins,p,q);startError=Math.max(startError,p.distanceTo(row.from));
  scatterMotion.sample(i,j,row.begins+row.duration,p,q);endError=Math.max(endError,p.distanceTo(row.to));
  for(const t of [row.begins,row.begins+row.duration]){
   scatterMotion.sample(i,j,t-eps,p0,q);scatterMotion.sample(i,j,t+eps,p1,q);maxBoundarySpeed=Math.max(maxBoundarySpeed,p0.distanceTo(p1)/(2*eps));
  }
 }
 return {startError,endError,maxBoundarySpeed,frames:DATA.length*2,shutterSeconds:SHUTTER_SECONDS,samples:BLUR_SAMPLES,holdSeconds:scatterMotion.start-13.22,scatterStart:scatterMotion.start,scatterEnd:scatterMotion.end,flowStart:14.5,handoffGapSeconds:14.5-scatterMotion.end,firstFlowStepPx:1440*startupProgress((1/30)/8.6)};
};
window.frameInfo=()=>({...frameState,hero:DATA.hero,atoms:DATA.atoms.length,source:DATA.source,savedStates:DATA.poses.length});
await document.fonts.ready;await ui.conDiagram.querySelector('img').decode();await Promise.all(galleryItems.map(v=>v.img.decode().catch(()=>{})));await window.renderFrame(Number(cfg.get('t')||0));window.ready=true;
let runId=0;
async function play(){const id=++runId,start=performance.now();async function tick(){if(id!==runId)return;const t=(performance.now()-start)/1000;await window.renderFrame(Math.min(t,DURATION));if(t<DURATION)requestAnimationFrame(tick);}tick();}
document.querySelector('#play').onclick=play;if(cfg.has('autoplay'))play();
