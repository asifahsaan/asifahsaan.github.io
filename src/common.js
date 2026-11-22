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
  const b=document.createElement('button');b.id='pdfbtn';b.className='btn';b.textContent='⬇ PDF';b.title='Preview this lab as a PDF, then download it';b.style.marginLeft='8px';b.onclick=openPdf;tabs.insertAdjacentElement('afterend',b);
  const p=document.createElement('button');p.id='pptxbtn';p.className='btn';p.textContent='⬇ PPTX';p.title='Make a PowerPoint deck for class (one slide per step)';p.style.marginLeft='6px';p.onclick=openPptx;b.insertAdjacentElement('afterend',p);}
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
/* ---------- PPTX export (PptxGenJS + html2canvas loaded on demand from a CDN) ---------- */
const PPTX_URL=['https://cdn.jsdelivr.net/npm/pptxgenjs@3.12.0/dist/pptxgen.bundle.js','https://unpkg.com/pptxgenjs@3.12.0/dist/pptxgen.bundle.js'],H2C_URL=['https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js','https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js'];
async function loadScript(urls){for(const u of [].concat(urls)){const ok=await new Promise(r=>{if(document.querySelector(`script[data-u="${u}"]`))return r(true);const s=document.createElement('script');s.src=u;s.dataset.u=u;s.onload=()=>r(true);s.onerror=()=>{s.remove();r(false);};document.head.appendChild(s);});if(ok)return;}throw new Error('Could not load the PowerPoint library');}
const SVGCSS='.halo{paint-order:stroke;stroke:#fff;stroke-width:3.2px;stroke-linejoin:round}.chipg rect{fill:#fff;fill-opacity:.94;stroke:#DEDED8;stroke-width:1}.chipg.gr rect{fill:#FDE8E6;stroke:#f0b9b3}.chipg.ac rect{stroke:#2B7A9B;stroke-width:1.8}.chipg text{font-size:10.5px;fill:#1E5873}.chipg.gr text{fill:#C0392B;font-weight:600}.chipg.ng text{fill:#B1500F}text{font-family:Arial,Helvetica,sans-serif}';
function svgToPng(svgEl,width){return new Promise((res,rej)=>{const c=svgEl.cloneNode(true);c.setAttribute('xmlns','http://www.w3.org/2000/svg');const vb=c.getAttribute('viewBox').split(/\s+/).map(Number),w=vb[2],h=vb[3];c.setAttribute('width',w);c.setAttribute('height',h);
  c.insertAdjacentHTML('afterbegin',`<style>${SVGCSS}</style>`);const img=new Image();
  img.onload=()=>{const sc=width/w,cv=document.createElement('canvas');cv.width=width;cv.height=Math.round(h*sc);const g=cv.getContext('2d');g.fillStyle='#fff';g.fillRect(0,0,cv.width,cv.height);g.drawImage(img,0,0,cv.width,cv.height);res({data:cv.toDataURL('image/png'),w:cv.width,h:cv.height});};
  img.onerror=()=>rej(new Error('Could not draw the diagram'));img.src='data:image/svg+xml;charset=utf-8,'+encodeURIComponent(new XMLSerializer().serializeToString(c));});}
let _pf=null;
async function pptxFrame(){if(_pf)return _pf;const f=document.createElement('iframe');f.style.cssText='position:fixed;left:0;top:0;width:1300px;height:2400px;opacity:0;pointer-events:none;border:0;z-index:-1';document.body.appendChild(f);const d=f.contentDocument;d.open();
  d.write('<!DOCTYPE html><html><head><meta charset="utf-8">'+[...document.querySelectorAll('head style')].map(s=>'<style>'+s.textContent+'</style>').join('')+'</head><body style="margin:0;padding:0;max-width:none;background:#fff"></body></html>');d.close();await new Promise(r=>setTimeout(r,400));_pf=f;return f;}
async function htmlToPng(html,wpx,scale){const f=await pptxFrame(),d=f.contentDocument,box=d.createElement('div');box.className='pptxbox';box.style.cssText=`width:${wpx}px;background:#fff;padding:6px 8px;box-sizing:border-box`;box.innerHTML=html;d.body.innerHTML='';d.body.appendChild(box);
  try{await d.fonts.ready;}catch(e){}const cv=await html2canvas(box,{scale:scale||1.6,backgroundColor:'#ffffff',logging:false});return{data:cv.toDataURL('image/png'),w:cv.width,h:cv.height};}
const plainTxt=s=>String(s||'').replace(/\$([^$]*)\$/g,'$1').replace(/\\(?:left|right|textcolor\{[^}]*\}|mathbf|text|small|quad|qquad|,|;|!)/g,' ').replace(/\\([a-zA-Z]+)/g,(m,a)=>({delta:'δ',sigma:'σ',tanh:'tanh',odot:'⊙',cdot:'·',times:'×',theta:'θ',eta:'η',partial:'∂',sum:'Σ',ln:'ln',max:'max',tilde:'~'}[a]||'')).replace(/[{}^_]/g,'').replace(/\*\*/g,'').replace(/\s+/g,' ').trim();
const fitBox=(img,bw,bh)=>{const r=Math.min(bw/img.w,bh/img.h);return{w:img.w*r,h:img.h*r};};
async function buildPptx(o,prog){await loadScript(PPTX_URL);await loadScript(H2C_URL);
  const pptx=new PptxGenJS();pptx.layout='LAYOUT_WIDE';pptx.title=document.title;pptx.author='Asif Ahsan';
  pptx.defineSlideMaster({title:'VL',background:{color:'FFFFFF'},objects:[{rect:{x:0,y:0,w:13.33,h:0.12,fill:{color:'2B7A9B'}}},{text:{text:'Visual Learning · Asif Ahsan · asifahsaan.github.io/visual-learning',options:{x:0.35,y:7.12,w:8,h:0.28,fontSize:9,color:'888888',fontFace:'Arial'}}}],slideNumber:{x:12.5,y:7.12,fontSize:9,color:'888888',fontFace:'Arial'}});
  const sel=steps.map((s,i)=>({s,i})).filter(({s})=>(s.ph==='S'&&o.setup)||(s.ph==='F'&&o.fwd)||(s.ph==='B'&&o.bwd)||((s.ph==='U'||s.ph==='C')&&o.upd));
  const total=sel.length+(o.concepts?(steps.gloss||[]).length:0)+(o.title?1:0);let done=0;const tick=m=>{done++;prog&&prog(done,total,m);};
  if(o.title){const sl=pptx.addSlide({masterName:'VL'});sl.addText(document.title.split('·')[0].trim(),{x:.6,y:1.2,w:12,h:1,fontSize:36,bold:true,color:'1E5873',fontFace:'Arial'});
    sl.addText('Step-by-step visual lecture',{x:.6,y:2.2,w:12,h:.5,fontSize:20,color:'555555',fontFace:'Arial'});
    const rows=Object.entries(S.cfg).filter(([k,v])=>k!=='tok'&&k!=='gran'&&!(v&&typeof v==='object'&&!Array.isArray(v))).slice(0,10).map(([k,v])=>[{text:k,options:{bold:true,color:'6b7280',fontSize:12}},{text:(Array.isArray(v)?JSON.stringify(v,(kk,x)=>typeof x==='number'?+x.toFixed(4):x):String(v)).slice(0,90),options:{fontSize:12}}]);
    if(rows.length)sl.addTable(rows,{x:.6,y:3.0,w:12,colW:[2.2,9.8],fontFace:'Arial',border:{type:'solid',color:'E5E7EB',pt:.5}});
    sl.addText('Asif Ahsan · Lecturer, Machine Learning & AI',{x:.6,y:6.5,w:12,h:.4,fontSize:14,color:'2B7A9B',fontFace:'Arial'});tick('title');}
  if(o.concepts)for(const c of (steps.gloss||[])){const sl=pptx.addSlide({masterName:'VL'});sl.addText(plainTxt(c.title),{x:.4,y:.3,w:12.5,h:.7,fontSize:26,bold:true,color:'1E5873',fontFace:'Arial'});
    sl.addText('CONCEPT',{x:.4,y:.18,w:4,h:.25,fontSize:10,color:'B58B00',bold:true,fontFace:'Arial'});
    const im=await htmlToPng(`<div class="lcard" style="border:0"><p style="font-size:20px"><b>What it is:</b> ${rt(c.what)}</p><p style="font-size:20px;margin-top:14px"><b>Why we use it:</b> ${rt(c.why)}</p></div>`,1150,1.6);
    const f=fitBox(im,12.4,5.6);sl.addImage({data:im.data,x:.45,y:1.3,w:f.w,h:f.h});tick('concept');}
  const saved=S.i;
  for(const {s,i} of sel){go(i);const sl=pptx.addSlide({masterName:'VL'});
    sl.addText(`STEP ${i+1} OF ${steps.length} · ${(PH_NAMES[s.ph]||'').toUpperCase()}`,{x:.4,y:.18,w:8,h:.25,fontSize:10,color:'6b7280',bold:true,fontFace:'Arial'});
    sl.addText(plainTxt(s.title),{x:.4,y:.42,w:12.5,h:.6,fontSize:22,bold:true,color:'1d2229',fontFace:'Arial'});
    sl.addText('Path: '+(s.path||''),{x:.4,y:.98,w:12.5,h:.25,fontSize:10,color:'6b7280',fontFace:'Arial'});
    if(o.diagram){const sv=document.querySelector('#svg');if(sv){const d=await svgToPng(sv,1500);const f=fitBox(d,7.95,4.05);sl.addImage({data:d.data,x:.35,y:1.3,w:f.w,h:f.h});}}
    const wk=await htmlToPng(`${s.ops&&s.ops.key&&s.ops.key.length?`<div class="opkey">${keyHTML(s.ops.key)}</div>`:''}<ul>${(s.what||[]).map(w=>`<li>${rt(w)}</li>`).join('')}</ul>`,o.diagram?500:1150,1.7);
    const wf=o.diagram?fitBox(wk,4.55,4.05):fitBox(wk,12.4,1.5);sl.addImage({data:wk.data,x:o.diagram?8.45:.45,y:o.diagram?1.3:1.4,w:wf.w,h:wf.h});
    const cb=await htmlToPng(`<div class="pres">${(s.res||[]).map(r=>K(r,true)).join('')}</div>`,1220,1.6),cf=fitBox(cb,12.5,o.diagram?1.8:3.7);
    const tall=o.diagram&&cb.h/cb.w*12.5>2.3;
    if(!tall)sl.addImage({data:cb.data,x:.4,y:o.diagram?5.35:3.3,w:cf.w,h:cf.h});else sl.addText('The calculation continues on the next slide →',{x:.4,y:5.5,w:12.5,h:.4,fontSize:14,italic:true,color:'2B7A9B',fontFace:'Arial'});
    sl.addNotes(`${plainTxt(s.title)}\n\n${(s.what||[]).map(plainTxt).join('\n')}\n\nTeaching note: ${plainTxt(s.say)}`);
    if(tall){const s2=pptx.addSlide({masterName:'VL'});s2.addText(`${plainTxt(s.title)} (calculation)`,{x:.4,y:.3,w:12.5,h:.6,fontSize:20,bold:true,color:'1d2229',fontFace:'Arial'});const f2=fitBox(cb,12.5,6.0);s2.addImage({data:cb.data,x:.4,y:1.1,w:f2.w,h:f2.h});}
    tick('step');}
  go(saved);if(_pf){_pf.remove();_pf=null;}return pptx;}
function openPptx(){const old=document.getElementById('pptxroot');if(old)old.remove();
  const cnt=ph=>steps.filter(s=>ph.includes(s.ph)).length,g=(steps.gloss||[]).length;
  const root=document.createElement('div');root.id='pptxroot';root.innerHTML=`<div class="pxbox"><h3>Make a PowerPoint for class</h3><div class="pxnote">One slide per step: title, the diagram of that step, the explanation and the calculation. Formulas and diagrams become pictures, titles and notes stay editable text. The teaching note goes into the speaker notes.</div>
    <div class="pxopts"><label><input type="checkbox" id="px-t" checked> Title slide with the example</label><label><input type="checkbox" id="px-c" checked> Concept slides (${g})</label><label><input type="checkbox" id="px-s" checked> Setup (${cnt('S')})</label><label><input type="checkbox" id="px-f" checked> Forward pass steps (${cnt('F')})</label><label><input type="checkbox" id="px-b" checked> Backward pass steps (${cnt('B')})</label><label><input type="checkbox" id="px-u" checked> Update and check (${cnt('UC')})</label><label><input type="checkbox" id="px-d" checked> Diagram on every slide</label></div>
    <div class="pxcount" id="px-n"></div><div class="pxbar"><div id="px-p"></div></div><div class="pxmsg" id="px-m"></div>
    <div class="pxbtns"><button class="btn pri" id="px-go">⬇ Build and download PPTX</button><button class="btn" id="px-x">Close</button></div><div class="pxnote" style="margin-top:8px">Needs an internet connection the first time (it loads the PowerPoint library). Building takes about a second per slide.</div></div>`;
  document.body.appendChild(root);const q=s=>root.querySelector(s);
  const opts=()=>({title:q('#px-t').checked,concepts:q('#px-c').checked,setup:q('#px-s').checked,fwd:q('#px-f').checked,bwd:q('#px-b').checked,upd:q('#px-u').checked,diagram:q('#px-d').checked});
  const est=()=>{const o=opts(),n=(o.title?1:0)+(o.concepts?g:0)+(o.setup?cnt('S'):0)+(o.fwd?cnt('F'):0)+(o.bwd?cnt('B'):0)+(o.upd?cnt('UC'):0);q('#px-n').textContent=`About ${n} slides`;};
  root.querySelectorAll('input').forEach(i=>i.onchange=est);est();
  q('#px-x').onclick=()=>root.remove();
  q('#px-go').onclick=async()=>{const b=q('#px-go');b.disabled=true;q('#px-m').textContent='Loading the PowerPoint library…';q('#px-p').style.width='2%';
    try{const pptx=await buildPptx(opts(),(d,t)=>{q('#px-p').style.width=Math.round(d/t*100)+'%';q('#px-m').textContent=`Slide ${d} of ${t}…`;});
      q('#px-m').textContent='Saving…';await pptx.writeFile({fileName:document.title.split('·')[0].trim().replace(/[^A-Za-z0-9 _-]/g,'')+' - Visual Learning.pptx'});q('#px-m').textContent='Done: the file was downloaded.';q('#px-p').style.width='100%';}
    catch(e){q('#px-m').textContent='Sorry, it did not work: '+e.message+' (check your internet connection and try again).';}b.disabled=false;};}
