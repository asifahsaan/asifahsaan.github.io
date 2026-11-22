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
