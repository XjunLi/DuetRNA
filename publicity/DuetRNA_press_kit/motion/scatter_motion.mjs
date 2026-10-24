import * as THREE from 'three';

const clamp=v=>Math.max(0,Math.min(1,v));
const seed=n=>{const x=Math.sin(n*127.1+311.7)*43758.5453;return x-Math.floor(x)};
// A compact, symmetric release with continuous velocity and acceleration.
// It reaches the first noise pose exactly as forward flow begins, without a settling hold.
export function travelProgress(u){u=clamp(u);return u*u*u*(10+u*(-15+6*u))}
export function startupProgress(u,span=.012){
 if(u<=0)return 0;if(u>=span)return Math.min(1,u);
 const v=u/span;return span*v*v*v*(6+v*(-8+3*v));
}

export function makeScatterMotion(poses,viewFrom,viewTo,rotation){
 const start=13.95,end=14.5,rows=[];
 const screenPosition=(a,view)=>new THREE.Vector3(...a.slice(0,3)).sub(view.center).applyQuaternion(rotation).multiplyScalar(view.scale);
 const screenRotation=a=>rotation.clone().multiply(new THREE.Quaternion(...a.slice(3)));
 for(let i=0;i<poses[0].length;i++){
  const pair=[];
  for(let j=0;j<2;j++){
   const n=i*2+j+1,a=poses.at(-1)[i][j],b=poses[0][i][j];
   const from=screenPosition(a,viewFrom),to=screenPosition(b,viewTo),distance=from.distanceTo(to);
   const tangent=new THREE.Vector3(seed(n+71)-.5,seed(n+113)-.5,seed(n+197)-.5).normalize();
   const radial=from.clone().normalize(),bend=radial.multiplyScalar(.7).addScaledVector(tangent,.6).normalize().multiplyScalar(Math.min(29,10+distance*.07));
   const delay=.025*seed(i+31)+.01*seed(n+17),duration=end-start-delay;
   pair.push({from,to,qFrom:screenRotation(a),qTo:screenRotation(b),bend,
    axis:tangent,twist:(seed(n+59)-.5)*.45,depth:(seed(n+139)-.5)*.16,
    begins:start+delay,duration});
  }
  rows.push(pair);
 }
 return {start,end,rows,axisLength:2.5*viewTo.scale,axisRadius:.15*viewTo.scale,
  sample(i,j,time,position,quaternion){
   const row=rows[i][j],u=clamp((time-row.begins)/row.duration),p=travelProgress(u);
   const bend=64*Math.pow(u,3)*Math.pow(1-u,3);
   position.copy(row.from).lerp(row.to,p).addScaledVector(row.bend,bend);
   quaternion.copy(row.qFrom).slerp(row.qTo,p);
   if(bend>0)quaternion.premultiply(new THREE.Quaternion().setFromAxisAngle(row.axis,row.twist*bend));
   return 1+row.depth*bend;
  },
 };
}
