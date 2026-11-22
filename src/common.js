/* ---------- shared helpers (Visual Learning labs) ---------- */
const R=String.raw, $=s=>document.querySelector(s);
const ACC='#2B7A9B', EDGE='#1E5873', ERR='#C0392B', GHOST='#D6D6D0';
const N=v=>{const a=Math.abs(v)<5e-5?0:v;return a.toFixed(4)};
const Q=v=>`(${N(v)})`;
const hl=v=>R`\textcolor{#1E5873}{\mathbf{${N(v)}}}`;
const cache=new Map();
function K(tex,disp){const key=(disp?'D':'I')+tex;if(cache.has(key))return cache.get(key);let h;
  try{h=katex.renderToString(tex,{displayMode:!!disp,throwOnError:false,strict:false})}catch(e){h=tex}cache.set(key,h);return h;}
const esc=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
function rt(s){return s.split(/(\$[^$]+\$)/).map(p=>(p.length>1&&p[0]==='$'&&p.endsWith('$'))?K(p.slice(1,-1)):esc(p).replace(/\*\*(.+?)\*\*/g,'<b>$1</b>')).join('');}
function mulberry(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;}}
function gauss(rnd){let u=0,v=0;while(!u)u=rnd();while(!v)v=rnd();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);}
const clone=o=>JSON.parse(JSON.stringify(o));
/* ---------- concept cards, colour pairing, activation formulas (shared by the labs) ---------- */
const PC={a:'#D9730D',b:'#7C3AED',bias:'#16A34A',c:'#0E7490',g:'#C0392B',loss:'#B7791F',dim:'#B9BFC6'};
const ccol=(col,tex)=>R`\textcolor{${col}}{${tex}}`;
let conceptsOpen=true;
function makeLearn(){const seen=new Set(),gloss=[];return{L:(id,title,what,why)=>{if(seen.has(id))return[];seen.add(id);const o={title,what,why};gloss.push(o);return[o];},gloss};}
const cardHTML=l=>`<div class="lcard"><div class="lh">${rt(l.title)}</div><p><b>What it is:</b> ${rt(l.what)}</p><p><b>Why we use it:</b> ${rt(l.why)}</p></div>`;
function conceptBlock(lr){return lr&&lr.length?`<details class="concepts" ${conceptsOpen?'open':''}><summary>Concept${lr.length>1?'s':''} in this step</summary>${lr.map(cardHTML).join('')}</details>`:'';}
function glossBlock(gloss,title){return`<details class="concepts" open><summary>${title||'All concepts used in this lab'}</summary><div class="cards">${(gloss||[]).map(cardHTML).join('')}</div></details>`;}
function keyHTML(key){return(key||[]).map(([c,t])=>`<span><i style="background:${c}"></i>${rt(t)}</span>`).join('');}
function initConcepts(){initPdf();const el=$('#learn');if(el)el.addEventListener('toggle',e=>{if(e.target.classList&&e.target.classList.contains('concepts'))conceptsOpen=e.target.open;},true);}
const ACTINFO2={
  tanh:{t:'Activation function: tanh',w:R`tanh is an S-shaped curve: $\tanh(z)=\dfrac{e^{z}-e^{-z}}{e^{z}+e^{-z}}$. Very negative $z$ gives about $-1$, $z=0$ gives $0$, very positive $z$ gives about $+1$.`,y:`Without a non-linear function, many layers collapse into one big linear map and the network could not learn curved, complex patterns. tanh also keeps every value between −1 and 1 and is centred on 0, which makes learning easier.`},
  sigmoid:{t:'Activation function: sigmoid',w:R`sigmoid is an S-shaped curve $\sigma(z)=\dfrac{1}{1+e^{-z}}$ that maps every number to a value between 0 and 1.`,y:`It adds the needed non-linearity and gives values that behave like probabilities (or like gates: 0 = closed, 1 = open). Drawback: its slope is at most 0.25, so errors shrink quickly when they travel back through many layers.`},
  relu:{t:'Activation function: ReLU',w:R`ReLU is $\max(0,z)$: negative values become 0, positive values pass unchanged.`,y:`It adds the needed non-linearity, is very cheap, and its slope is exactly 1 for positive $z$. Drawback: no upper limit, and a unit that is stuck below 0 passes no gradient.`}};
/* two lines: formula, then the numbers; lhs = left side symbol, sym = the argument symbol */
function actFormulaLines(act,sv,hv,lhs,sym){
  if(act==='tanh'){const a=Math.exp(sv),b=Math.exp(-sv);return[R`${lhs}=\tanh(${sym})=\dfrac{e^{${sym}}-e^{-${sym}}}{e^{${sym}}+e^{-${sym}}}`,R`=\dfrac{e^{${N(sv)}}-e^{${N(-sv)}}}{e^{${N(sv)}}+e^{${N(-sv)}}}=\dfrac{${N(a)}-${N(b)}}{${N(a)}+${N(b)}}=\dfrac{${N(a-b)}}{${N(a+b)}}=${hl(hv)}`];}
  if(act==='sigmoid')return[R`${lhs}=\sigma(${sym})=\dfrac{1}{1+e^{-${sym}}}`,R`=\dfrac{1}{1+e^{${N(-sv)}}}=\dfrac{1}{1+${N(Math.exp(-sv))}}=${hl(hv)}`];
  return[R`${lhs}=\max(0,${sym})`,R`=\max(0,${N(sv)})=${hl(hv)}`];}
function curveSVG(kind,col){const w=46,h=26;let body='';
  if(kind==='softmax'){const hs=[.35,.8,.45];body=hs.map((v,i)=>`<rect x="${6+i*13}" y="${h-3-v*(h-8)}" width="9" height="${v*(h-8)}" fill="${col}" opacity=".8"/>`).join('')+`<line x1="3" y1="${h-3}" x2="${w-3}" y2="${h-3}" stroke="#999" stroke-width="1"/>`;}
  else{const f={tanh:Math.tanh,sigmoid:z=>1/(1+Math.exp(-z)),relu:z=>Math.max(0,z),identity:z=>z,linear:z=>z,none:z=>z}[kind]||(z=>z);const lo=kind==='tanh'?-1:kind==='sigmoid'?0:kind==='relu'?0:-3,hi=kind==='tanh'?1:kind==='sigmoid'?1:3;
    const pts=Array.from({length:25},(_,i)=>{const z=-3+6*i/24;return`${(3+(w-6)*i/24).toFixed(1)},${(h-3-(h-6)*(f(z)-lo)/(hi-lo)).toFixed(1)}`;}).join(' ');
    body=`<line x1="3" y1="${h-3}" x2="${w-3}" y2="${h-3}" stroke="#bbb"/><line x1="${w/2}" y1="2" x2="${w/2}" y2="${h-2}" stroke="#ddd"/><polyline points="${pts}" fill="none" stroke="${col}" stroke-width="2"/>`;}
  return`<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" style="vertical-align:middle;background:#fff;border:1px solid #DEDED8;border-radius:5px">${body}</svg>`;}
/* ---------- PDF export: preview first, then "Download PDF" (uses the browser's Save as PDF) ---------- */
function initPdf(){if(document.getElementById('pdfbtn'))return;const tabs=document.querySelector('header .tabs');if(!tabs)return;
  const b=document.createElement('button');b.id='pdfbtn';b.className='btn';b.textContent='⬇ PDF';b.title='Preview this lab as a PDF, then download it';b.style.marginLeft='8px';b.onclick=openPdf;tabs.insertAdjacentElement('afterend',b);}
const PH_NAMES={S:'Setup',F:'Forward',B:'Backward',U:'Update',C:'Check'};
function cfgRows(cfg){const rows=[];for(const [k,v] of Object.entries(cfg)){if(k==='tok'||k==='gran')continue;if(v&&typeof v==='object'&&!Array.isArray(v))continue;
  const t=Array.isArray(v)?JSON.stringify(v,(kk,x)=>typeof x==='number'?+x.toFixed(4):x).slice(0,140):String(v);rows.push(`<tr><td>${esc(k)}</td><td>${esc(t)}</td></tr>`);}return rows.join('');}
function pdfDocHTML(o,diagrams){const cfg=S.cfg,title=document.title;
  let h=`<h1>${esc(title)}</h1><div class="psub">Visual Learning · Asif Ahsan · asifahsaan.github.io/visual-learning</div>`;
  h+=`<h2>The example</h2><table class="pt">${cfgRows(cfg)}</table>`;
  if(o.concepts&&steps.gloss&&steps.gloss.length)h+=`<h2>Concepts</h2>`+steps.gloss.map(cardHTML).join('');
  if(o.calc){h+=`<h2>Step-by-step calculations (${steps.length} steps)</h2>`;
    steps.forEach((s,i)=>{h+=`<div class="pstep"><div class="pk">Step ${i+1} of ${steps.length} · ${PH_NAMES[s.ph]||''}</div><div class="pt2">${rt(s.title)}</div><div class="pp">Path: ${esc(s.path||'')}</div>`
      +(diagrams&&diagrams[i]?`<div class="psvg">${diagrams[i]}</div>`:'')
      +(s.ops&&s.ops.key&&s.ops.key.length?`<div class="opkey">${keyHTML(s.ops.key)}</div>`:'')
      +`<ul>${(s.what||[]).map(w=>`<li>${rt(w)}</li>`).join('')}</ul><div class="pres">${(s.res||[]).map(r=>K(r,true)).join('')}</div>`+(s.say?`<div class="psay"><b>Teaching note:</b> ${rt(s.say)}</div>`:'')+`</div>`;});}
  return`<div class="pdoc">${h}</div>`;}
function openPdf(){const old=document.getElementById('pdfroot');if(old)old.remove();
  const root=document.createElement('div');root.id='pdfroot';
  root.innerHTML=`<div id="pdfbar"><b>PDF preview</b><label><input type="checkbox" id="po-c" checked> Concepts</label><label><input type="checkbox" id="po-k" checked> Calculations</label><label><input type="checkbox" id="po-d" ${steps.length<=70?'checked':''}> Diagram with every step</label><span class="pdfnote">Check the preview. If it looks right, press Download and choose “Save as PDF”.</span><span style="flex:1"></span><button class="btn pri" id="po-dl">⬇ Download PDF</button><button class="btn" id="po-x">Close</button></div><div id="pdfscroll"><div id="pdfpage"><div class="pdoc">Preparing the preview…</div></div></div>`;
  document.body.appendChild(root);
  const page=root.querySelector('#pdfpage');let busy=false;
  const render=()=>{if(busy)return;busy=true;page.innerHTML='<div class="pdoc">Preparing the preview…</div>';
    setTimeout(()=>{const o={concepts:root.querySelector('#po-c').checked,calc:root.querySelector('#po-k').checked,diag:root.querySelector('#po-d').checked};let diagrams=null;
      if(o.diag&&o.calc&&typeof go==='function'){const saved=S.i;diagrams=[];for(let i=0;i<steps.length;i++){go(i);const sv=document.querySelector('#svg');diagrams.push(sv?sv.outerHTML.replace(/id="([^"]+)"/g,(m,a)=>`id="${a}__${i}"`).replace(/url\(#([^)]+)\)/g,(m,a)=>`url(#${a}__${i})`):'');}go(saved);}
      page.innerHTML=pdfDocHTML(o,diagrams);
      page.querySelectorAll('.pres').forEach(p=>{const cw=p.clientWidth;p.querySelectorAll('.katex-display').forEach(d=>{const k=d.querySelector('.katex');const w=k?k.getBoundingClientRect().width:0;if(w>cw+2)d.style.zoom=Math.max(.45,cw/w);});});
      busy=false;},30);};
  ['#po-c','#po-k','#po-d'].forEach(s=>root.querySelector(s).onchange=render);
  root.querySelector('#po-x').onclick=()=>root.remove();
  root.querySelector('#po-dl').onclick=()=>{const t=document.title;document.title=t.split('·')[0].trim().replace(/[^A-Za-z0-9 _-]/g,'')+' - Visual Learning';window.addEventListener('afterprint',()=>{document.title=t;},{once:true});window.print();};
  document.addEventListener('keydown',function esc_(e){if(e.key==='Escape'&&document.getElementById('pdfroot')){document.getElementById('pdfroot').remove();document.removeEventListener('keydown',esc_);}});
  render();}
