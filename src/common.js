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
function initConcepts(){const el=$('#learn');if(el)el.addEventListener('toggle',e=>{if(e.target.classList&&e.target.classList.contains('concepts'))conceptsOpen=e.target.open;},true);}
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
