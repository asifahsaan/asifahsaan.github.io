/* ---------- floating explainer + draggable book icon ---------- */
const panel=$('#panel'),fhead=$('#fhead'),fab=$('#fab');
let panelOpen=true,fabPos=null;const FS=58,FM=22;
const fabDefault=()=>({x:innerWidth-FS-FM,y:innerHeight-FS-FM});
const clampFab=p=>({x:Math.max(6,Math.min(innerWidth-FS-6,p.x)),y:Math.max(6,Math.min(innerHeight-FS-6,p.y))});
function placeFab(){const p=clampFab(fabPos||fabDefault());fab.style.left=p.x+'px';fab.style.top=p.y+'px';}
function syncPanel(){const on=tab==='step';panel.style.display=(on&&panelOpen)?'':'none';fab.style.display=(on&&!panelOpen)?'':'none';}
function openPanel(){panelOpen=true;const w=Math.min(540,innerWidth*.94);if(!panel.style.width)panel.style.width=w+'px';if(!panel.style.height)panel.style.height=Math.round(Math.min(560,innerHeight*.84))+'px';
  panel.style.visibility='hidden';syncPanel();const pw=panel.offsetWidth,ph=panel.offsetHeight;
  const f=clampFab(fabPos||fabDefault()),cx=f.x+FS/2,cy=f.y+FS/2;
  let left=cx>innerWidth/2?f.x+FS-pw:f.x,top=cy>innerHeight/2?f.y+FS-ph:f.y;
  left=Math.max(8,Math.min(innerWidth-pw-8,left));top=Math.max(8,Math.min(innerHeight-ph-8,top));
  panel.style.left=left+'px';panel.style.top=top+'px';panel.style.transformOrigin=`${cx-left}px ${cy-top}px`;panel.style.visibility='';
  panel.classList.remove('closing');panel.classList.add('opening');setTimeout(()=>panel.classList.remove('opening'),420);}
function closePanel(){const r=panel.getBoundingClientRect(),f=clampFab(fabPos||fabDefault());
  panel.style.transformOrigin=`${f.x+FS/2-r.left}px ${f.y+FS/2-r.top}px`;panel.classList.add('closing');
  setTimeout(()=>{panelOpen=false;panel.classList.remove('closing');placeFab();syncPanel();},230);}
fab.addEventListener('pointerdown',e=>{const r=fab.getBoundingClientRect(),ox=e.clientX-r.left,oy=e.clientY-r.top,sx=e.clientX,sy=e.clientY;let moved=false;
  fab.setPointerCapture(e.pointerId);
  const mv=ev=>{if(Math.hypot(ev.clientX-sx,ev.clientY-sy)>5&&!moved){moved=true;fab.classList.add('drag');}if(moved){fabPos=clampFab({x:ev.clientX-ox,y:ev.clientY-oy});placeFab();}};
  const up=()=>{fab.classList.remove('drag');fab.removeEventListener('pointermove',mv);fab.removeEventListener('pointerup',up);fab.removeEventListener('pointercancel',up);if(!moved)openPanel();};
  fab.addEventListener('pointermove',mv);fab.addEventListener('pointerup',up);fab.addEventListener('pointercancel',up);});
fhead.addEventListener('pointerdown',e=>{if(e.target.closest('button'))return;const r=panel.getBoundingClientRect(),ox=e.clientX-r.left,oy=e.clientY-r.top;
  fhead.setPointerCapture(e.pointerId);panel.classList.add('drag');
  const mv=ev=>{panel.style.left=Math.max(8-panel.offsetWidth+120,Math.min(innerWidth-120,ev.clientX-ox))+'px';panel.style.top=Math.max(8,Math.min(innerHeight-44,ev.clientY-oy))+'px';};
  const up=()=>{panel.classList.remove('drag');fhead.removeEventListener('pointermove',mv);fhead.removeEventListener('pointerup',up);fhead.removeEventListener('pointercancel',up);};
  fhead.addEventListener('pointermove',mv);fhead.addEventListener('pointerup',up);fhead.addEventListener('pointercancel',up);});
addEventListener('resize',()=>{placeFab();if(!panelOpen)return;const r=panel.getBoundingClientRect();
  if(r.right>innerWidth)panel.style.left=Math.max(8,innerWidth-r.width-8)+'px';if(r.top>innerHeight-44)panel.style.top=Math.max(8,innerHeight-120)+'px';});
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&panelOpen&&tab==='step'&&panel.style.display!=='none')closePanel();});