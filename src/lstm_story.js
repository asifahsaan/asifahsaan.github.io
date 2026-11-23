/* ================= Story tab: biryani -> pasta, memory slots with real LSTM gates ================= */
const SV=['Asif','Ahmed','eats','loves','biryani','pasta','daily','?'];
const SU=[{n:'Biryani',s:'Pakistani food',c:'#D9730D'},{n:'Pasta',s:'Italian food',c:'#7C3AED'},{n:'Asif',s:'who is talking',c:'#0E7490'},{n:'Ahmed',s:'who is talking',c:'#16A34A'}];
const STORIES={'Deck story: Asif eats biryani… Ahmed loves pasta… ?':['Asif','eats','biryani','Ahmed','loves','pasta','?'],
 'Only Asif: biryani every day':['Asif','eats','biryani','daily','eats','biryani','?'],
 'Reverse: Ahmed first, then Asif':['Ahmed','loves','pasta','Asif','eats','biryani','?'],
 'Ahmed never gets a dish':['Asif','eats','biryani','Ahmed','loves','daily','?']};
let ST={u:null,words:[...STORIES[Object.keys(STORIES)[0]]],t:0,play:null,m:null,cfg:null};
function storyCfg(words){const T=words.length,d=SV.length,H=4,ix=w=>SV.indexOf(w),Z=()=>Array.from({length:H},()=>Array(d).fill(0)),U0=()=>Array.from({length:H},()=>Array(H).fill(0));
  const W={f:Z(),i:Z(),g:Z(),o:Z()},b={f:[6,6,6,6],i:[-6,-6,-6,-6],g:[0,0,0,0],o:[-5,-5,-5,-5]};
  W.f[0][ix('Ahmed')]=-12;W.f[2][ix('Ahmed')]=-12;W.f[1][ix('Asif')]=-12;W.f[3][ix('Asif')]=-12;
  W.i[0][ix('biryani')]=12;W.i[1][ix('pasta')]=12;W.i[2][ix('Asif')]=12;W.i[3][ix('Ahmed')]=12;
  W.g[0][ix('biryani')]=3;W.g[1][ix('pasta')]=3;W.g[2][ix('Asif')]=3;W.g[3][ix('Ahmed')]=3;
  for(let u=0;u<H;u++)W.o[u][ix('?')]=10;
  const p={};for(const k of GI)p[k]={W:W[k],U:U0(),b:b[k]};
  return{cell:'lstm',T,d,H,C:2,out:'softmax',mode:'last',eta:0.5,p,Why:[[3,-3,0,0],[-3,3,0,0]],by:[0,0],x:words.map(w=>SV.map((_,j)=>j===ix(w)?1:0)),t:[[0,1]],gran:'vector'};}
const CAST=[
 {e:'🧺',n:'Mixer',f:'W·[h, x] + b',d:'Mixes the new word with what was just said and adds a bias, one mixer per gate.'},
 {e:'🎚️',n:'Dimmer switch (σ)',f:'σ(z)',d:'Turns any number into a dial between 0 (off) and 1 (fully on). Used by the three gates.'},
 {e:'💭',n:'Dreamer (tanh)',f:'c̃ = tanh(z)',d:'Dreams up new content to remember, as a number between −1 and 1.'},
 {e:'💂',n:'Forget Guard (×f)',f:'f ⊙ c',d:'Stands on the belt. Multiplies old memory by f: 0 wipes it, 1 lets it pass.'},
 {e:'✍️',n:'Scribe (×i)',f:'i ⊙ c̃',d:'Writes the dream onto the belt, but only as much as the input dial i allows.'},
 {e:'🧺➕',n:'Basket (+)',f:'c = kept + written',d:'Puts the kept memory and the new writing together on the belt.'},
 {e:'🚚',n:'Conveyor belt',f:'c(t−1) → c(t)',d:'The long-term memory. It carries everything from word to word.'},
 {e:'🗜️',n:'Squeezer (tanh)',f:'tanh(c)',d:'Squashes the memory into −1…1 before anyone looks at it.'},
 {e:'📢',n:'Speaker (×o)',f:'h = o ⊙ tanh(c)',d:'Decides how much of the memory is said out loud as the answer h.'}];
function storyOps(t,j){const m=ST.m,cfg=ST.cfg,P=cfg.p,x=cfg.x[t],hp=m.h[t],G=m.G[t],cp=m.c[t][j],ct=m.c[t+1][j],z={};GI.forEach(k=>z[k]=preAct(P,k,x,hp)[j]);
  const th=Math.tanh(ct),kept=G.f[j]*cp,wr=G.i[j]*G.g[j];
  return[[0,`Mixes “${ST.words[t]}” and the last output for each gate`,`z_f=${f2(z.f)}, z_i=${f2(z.i)}, z_c=${f2(z.g)}, z_o=${f2(z.o)}`],
   [1,`Turns the three gate numbers into dials`,`f=σ(${f2(z.f)})=${f2(G.f[j])}, i=σ(${f2(z.i)})=${f2(G.i[j])}, o=σ(${f2(z.o)})=${f2(G.o[j])}`],
   [2,`Dreams up new content`,`c̃=tanh(${f2(z.g)})=${f2(G.g[j])}`],
   [3,G.f[j]<.2?`Wipes the old memory!`:G.f[j]>.8?`Lets the old memory pass`:`Lets part of the old memory pass`,`${f2(G.f[j])} × ${f2(cp)} = ${f2(kept)}`],
   [4,G.i[j]>.8?`Writes the new idea`:`Writes almost nothing`,`${f2(G.i[j])} × ${f2(G.g[j])} = ${f2(wr)}`],
   [5,`Puts kept + written together on the belt`,`${f2(kept)} + ${f2(wr)} = ${f2(ct)}`],
   [7,`Squashes the new memory`,`tanh(${f2(ct)}) = ${f2(th)}`],
   [8,G.o[j]>.5?`Speaks!`:`Stays quiet`,`${f2(G.o[j])} × ${f2(th)} = ${f2(G.o[j]*th)}`]];}
function storyRun(){ST.cfg=storyCfg(ST.words);ST.m=forward(ST.cfg);const m=ST.m,cfg=ST.cfg;
  ST.pred=m.h.slice(1).map(h=>{const u=mv(cfg.Why,h).map((v,q)=>v+cfg.by[q]),mx=Math.max(...u),e=u.map(v=>Math.exp(v-mx)),s=e[0]+e[1];return e.map(v=>v/s);});}
const sbar=(v,col)=>{const w=Math.min(1,Math.abs(v))*50;return`<div class="sbar"><i class="mid"></i><b style="${v>=0?`left:50%;width:${w}%`:`right:50%;width:${w}%`};background:${col}"></b></div>`;};
const gbar=(v,col)=>`<div class="gbar"><b style="width:${Math.round(v*100)}%;background:${col}"></b></div>`;
function storyVerdict(t){const m=ST.m,G=m.G[t],w=ST.words[t],L=[];
  SU.forEach((u,j)=>{const before=m.c[t][j],f=G.f[j],i=G.i[j],g=G.g[j];
    if(before>0.3&&f<0.2)L.push(`The forget gate **closes** on **${u.n}** (f = ${f2(f)}): the memory of ${u.n} (${f2(before)}) is erased.`);
    if(i>0.8&&Math.abs(g)>0.5)L.push(`The input gate **opens** for **${u.n}** (i = ${f2(i)}) and writes ${f2(i*g)} into that slot.`);});
  if(Math.max(...G.o)>0.8)L.push(`The output gate **opens** (o ≈ ${f2(Math.max(...G.o))}): the memory is revealed as the hidden state and the network answers.`);
  if(!L.length)L.push(`“${w}” changes nothing important: gates stay as they are, so the memory is **kept** (forget ≈ 1, input ≈ 0).`);return L;}
function renderStory(){
  const root=$('#storyroot');
  if(S.cfg.cell!=='lstm'){root.innerHTML=`<div class="note" style="padding:18px">The story uses the <b>LSTM</b> memory cell. You are on GRU. <button class="btn pri" id="storygo">Switch to LSTM</button></div>`;$('#storygo').onclick=()=>$('#cellsel button[data-c=lstm]').click();return;}
  storyRun();const m=ST.m,T=ST.cfg.T,t=Math.min(ST.t,T-1);ST.t=t;const G=m.G[t],pr=ST.pred[t],fin=t===T-1;const ch=SU.map((u,j)=>Math.abs(m.c[t+1][j]-m.c[t][j]));const er=SU.map((u,j)=>m.c[t][j]>.3&&G.f[j]<.2?j:-1).filter(j=>j>=0);const uj=ST.u!=null?ST.u:er.length?er[0]:ch.indexOf(Math.max(...ch));const ops=storyOps(t,uj);
  const hasMemory=j=>true;
  const sent=ST.words.map((w,k)=>`<button class="wchip${k===t?' on':''}${k>t?' fut':''}" data-k="${k}">${esc(w)}</button>`).join('');
  const sel=ST.words.map((w,k)=>`<select data-k="${k}">${SV.map(v=>`<option${v===w?' selected':''}>${v}</option>`).join('')}</select>`).join('');
  const slots=SU.map((u,j)=>{const c0=m.c[t][j],f=G.f[j],i=G.i[j],g=G.g[j],kept=f*c0,wr=i*g,c1=m.c[t+1][j],chg=Math.abs(c1-c0)>0.05;
    return`<div class="slot${chg?' chg':''}${j===uj?' pick':''}" data-j="${j}" style="--sc:${u.c}"><div class="sh"><b>${u.n}</b><span>${u.s}</span></div>
      <div class="srow"><span>memory before</span>${sbar(c0,u.c)}<em>${f2(c0)}</em></div>
      <div class="srow"><span>× forget f</span>${gbar(f,'#E0A100')}<em>${f2(f)}</em></div>
      <div class="srow"><span>= kept</span>${sbar(kept,u.c)}<em>${f2(kept)}</em></div>
      <div class="srow"><span>+ input i × new g̃</span>${gbar(i,'#E0A100')}<em>${f2(i)}×${f2(g)}</em></div>
      <div class="srow"><span>= written</span>${sbar(wr,u.c)}<em>${f2(wr)}</em></div>
      <div class="srow fin"><span>memory after</span>${sbar(c1,u.c)}<em>${f2(c1)}</em></div>
      <div class="srow"><span>output gate o</span>${gbar(G.o[j],'#E0A100')}<em>${f2(G.o[j])}</em></div></div>`;}).join('');
  const tl=`<table class="tline"><tr><th></th>${ST.words.map((w,k)=>`<th class="${k===t?'on':''}" data-k="${k}">${esc(w)}</th>`).join('')}</tr>${SU.map((u,j)=>`<tr><th style="color:${u.c}">${u.n}</th>${ST.words.map((w,k)=>{const v=m.c[k+1][j],a=Math.min(1,Math.abs(v));return`<td class="${k===t?'on':''}" data-k="${k}" style="background:${v>=0?`rgba(${hex(u.c)},${a*.85})`:`rgba(120,120,120,${a*.6})`};color:${a>.55?'#fff':'#222'}">${f2(v)}</td>`}).join('')}</tr>`).join('')}</table>`;
  const ans=pr[1]>pr[0]?'Italian cuisine':'Pakistani cuisine';
  root.innerHTML=`<div class="storytop"><div class="seg" id="s-pre">${Object.keys(STORIES).map((n,k)=>`<button data-n="${esc(n)}">${esc(n.split(':')[0])}</button>`).join('')}</div></div>
   <div class="story-sent">${sent}</div>
   <div class="story-ctrl"><button class="btn" id="s-prev">◀ Prev</button><button class="btn pri" id="s-next">Next ▶</button><button class="btn" id="s-play">▶ Play</button>
     <details class="edit"><summary>Edit the sentence</summary><div class="sel">${sel}</div></details>
     <button class="btn" id="s-arch">See this step in the Architecture tab</button></div>
   <div class="verdict"><div class="vh">Step ${t+1} of ${T}: reading “${esc(ST.words[t])}”</div>${storyVerdict(t).map(l=>`<div>${rt(l)}</div>`).join('')}</div>
   <div class="slots">${slots}</div>
   <h3 class="sh3">Who did what at this step? <span class="note">Following the <b style="color:${SU[uj].c}">${SU[uj].n}</b> slot (click another slot to follow it)</span></h3>
   <ol class="ops">${ops.map(([c,a,nu],n)=>`<li><span class="oe">${CAST[c].e}</span><div><b>${n+1}. ${CAST[c].n}</b>: ${esc(a)}<div class="on">${esc(nu)}</div></div></li>`).join('')}</ol>
   <details class="cast" open><summary>The cast: every part of the LSTM as a character</summary><div class="castg">${CAST.map(c=>`<div class="cc"><div class="ce">${c.e}</div><div><b>${c.n}</b><div class="cf">${esc(c.f)}</div><div>${esc(c.d)}</div></div></div>`).join('')}</div></details>
   <h3 class="sh3">Memory over the whole sentence</h3>${tl}
   <div class="predrow"><div><b>Prediction if asked now</b> <span class="note">(${Math.max(...G.o)>0.5?'output gate open':'output gate closed, so the answer is not meaningful yet'})</span></div>
     <div class="pbars"><span>Pakistani</span>${gbar(pr[0],'#D9730D')}<em>${(pr[0]*100).toFixed(1)}%</em><span>Italian</span>${gbar(pr[1],'#7C3AED')}<em>${(pr[1]*100).toFixed(1)}%</em></div>
     ${fin?`<div class="ans">Answer: <b>${ans}</b></div>`:''}</div>
   <div class="note" style="margin-top:10px">All numbers are real LSTM gates ($f,i,\\tilde c,o$) computed in your browser. The weights are <b>set by hand</b> to teach the idea (Ahmed makes the Biryani slot forget, pasta writes the Pasta slot), not trained. The story slides from the lecture are shown with real arithmetic.</div>`;
  root.querySelectorAll('.note').forEach(el=>{if(el.textContent.includes('$'))el.innerHTML=rt(el.textContent);});
  root.querySelectorAll('#s-pre button').forEach(b=>{b.classList.toggle('on',JSON.stringify(STORIES[b.dataset.n])===JSON.stringify(ST.words));b.onclick=()=>{ST.words=[...STORIES[b.dataset.n]];ST.t=0;stopStory();renderStory();};});
  root.querySelectorAll('.slot').forEach(el=>el.onclick=()=>{ST.u=+el.dataset.j;renderStory();});
  root.querySelectorAll('.wchip,.tline th[data-k],.tline td').forEach(el=>el.onclick=()=>{ST.t=+el.dataset.k;ST.u=null;stopStory();renderStory();});
  root.querySelectorAll('.sel select').forEach(s=>s.onchange=()=>{ST.words[+s.dataset.k]=s.value;renderStory();root.querySelector('details.edit').open=true;});
  $('#s-prev').onclick=()=>{stopStory();ST.t=Math.max(0,ST.t-1);ST.u=null;renderStory();};$('#s-next').onclick=()=>{stopStory();ST.t=Math.min(T-1,ST.t+1);ST.u=null;renderStory();};
  $('#s-play').textContent=ST.play?'⏸ Pause':'▶ Play';
  $('#s-play').onclick=()=>{if(ST.play){stopStory();renderStory();return;}if(ST.t>=T-1)ST.t=0;ST.play=setInterval(()=>{if(ST.t>=T-1){stopStory();renderStory();return;}ST.t++;ST.u=null;renderStory();},1800);renderStory();};
  $('#s-arch').onclick=()=>{const keep=ST.t;S.cfg=clone(ST.cfg);S.i=0;TR.cfg=null;A.open.clear();A.t=keep;A.sel=null;A.zoom=null;rebuild(true);setTab('arch');};
}
function stopStory(){if(ST.play){clearInterval(ST.play);ST.play=null;}}
function hex(h){return[1,3,5].map(i=>parseInt(h.slice(i,i+2),16)).join(',');}
