/* ---------- "Look inside" views ---------- */
const RIN=new Set(['mx','mh','sum','act','mo','oact','loss']);
const UC=['#D9730D','#7C3AED','#16A34A','#0E7490','#B7791F','#C0392B'];
const uvec=v=>v.map((x,j)=>`<span class="uv" style="--uc:${UC[j%6]}"><i>unit ${j+1}</i>${N(x)}</span>`).join('');
function inCurve(kind,zs,ys){const W=360,Hh=200,p=34,xr=kind==='sig'?[-6,6]:[-3,3],yr=kind==='sig'?[0,1]:kind==='relu'?[0,3]:[-1,1];
  const X=z=>p+(Math.max(xr[0],Math.min(xr[1],z))-xr[0])/(xr[1]-xr[0])*(W-2*p),Y=y=>Hh-p-(Math.max(yr[0],Math.min(yr[1],y))-yr[0])/(yr[1]-yr[0])*(Hh-2*p);
  const f=kind==='sig'?z=>1/(1+Math.exp(-z)):kind==='relu'?z=>Math.max(0,z):Math.tanh;let d='';for(let i=0;i<=120;i++){const z=xr[0]+(xr[1]-xr[0])*i/120;d+=(i?'L':'M')+X(z).toFixed(1)+' '+Y(f(z)).toFixed(1);}
  const xt=kind==='sig'?[-6,-3,0,3,6]:[-3,-1.5,0,1.5,3],ticks=xt.map(z=>`<text x="${X(z)}" y="${Hh-p+16}" text-anchor="middle" class="tk">${z}</text>`).join('');
  const yv=kind==='sig'?[0,.5,1]:kind==='relu'?[0,1,2,3]:[-1,0,1],yt=yv.map(y=>`<line x1="${p}" x2="${W-p}" y1="${Y(y)}" y2="${Y(y)}" class="gl2"/><text x="${p-6}" y="${Y(y)+4}" text-anchor="end" class="tk">${y}</text>`).join('');
  const b=Y(yr[0]);
  const dots=zs.map((z,j)=>{const c=UC[j%6],x=X(z),yy=Y(ys[j]);return`<line x1="${x}" x2="${x}" y1="${b}" y2="${b}" stroke="${c}" stroke-width="2" stroke-dasharray="4 3"><animate attributeName="y2" from="${b}" to="${yy}" dur="1.6s" begin=".9s" fill="freeze"/></line>
   <line x1="${x}" x2="${x}" y1="${yy}" y2="${yy}" stroke="${c}" stroke-width="2" stroke-dasharray="4 3"><animate attributeName="x2" from="${x}" to="${p}" dur="1.2s" begin="2.5s" fill="freeze"/></line>
   <circle cx="${x}" cy="${b}" r="6.5" fill="${c}" stroke="#fff" stroke-width="2"><animate attributeName="cy" from="${b}" to="${yy}" dur="1.6s" begin=".9s" fill="freeze"/></circle>
   <circle cx="${p}" cy="${yy}" r="0" fill="${c}" stroke="#fff" stroke-width="2"><animate attributeName="r" from="0" to="6.5" dur=".4s" begin="3.7s" fill="freeze"/></circle>`}).join('');
  return`<svg class="icurve" viewBox="0 0 ${W} ${Hh}">${yt}<line x1="${p}" x2="${W-p}" y1="${Hh-p}" y2="${Hh-p}" class="ax"/><line x1="${p}" x2="${p}" y1="${p-8}" y2="${Hh-p}" class="ax"/>${ticks}<path d="${d}" fill="none" stroke="#475569" stroke-width="2.5"/>${dots}<text x="${W-p}" y="${p-12}" text-anchor="end" class="tk">number in (bottom) → number out (left)</text></svg>`;}
function rCard(title,sym,symc,lede,inHTML,fn,outHTML,curve,table){
  return`<div class="ihead"><span class="isym" style="background:${symc};color:${symc===AC.box?'#3C5068':'#fff'}">${sym}</span><div><h3>${title}</h3><div class="ilede">${esc(lede)}</div></div></div>
  <div class="iflow"><div class="ibox"><div class="ik">What came in</div>${inHTML}</div><div class="iarr">➜</div><div class="ibox mid"><div class="ik">What happens inside</div>${fn}</div><div class="iarr">➜</div><div class="ibox"><div class="ik">What came out</div>${outHTML}</div></div>
  ${curve?`<div class="icwrap">${curve}</div>`:''}<div class="itw">${table}</div>`;
}
function rInside(nm,t){
  const cfg=S.cfg,m=RA.m,x=cfg.x[t],hp=m.h[t],ht=m.h[t+1],sv=m.s[t],H=cfg.H,AF=ACTS[cfg.act],oh=oHere(t),k=outSteps(cfg).indexOf(t);
  const T=(h,r)=>`<table class="itab"><tr>${h.map(c=>`<th>${c}</th>`).join('')}</tr>${r.map((row,i)=>`<tr style="--i:${i}">${row.map(c=>`<td>${c}</td>`).join('')}</tr>`).join('')}</table>`;
  const dot=j=>`<b class="ud" style="background:${UC[j%6]}">${j+1}</b>`;
  const prod=(row,v)=>{const sm=row.reduce((s,w,i)=>s+w*v[i],0);return`${row.map((w,i)=>`${f2(w)}×${f2(v[i])}`).join(' + ')}<br><b>= ${N(sm)}</b>`;};
  if(nm==='mx'){const xa=mv(cfg.Wxh,x);return rCard('Inside the input weights W_xh · x','W·x',AC.op,'Every hidden unit has its own row of weights. It multiplies each input number by its weight and adds the results.',`<span class="vl">x(t)</span>${uvec(x)}`,K(R`(W_{xh}x)_j=\sum_i W_{xh,ji}\,x_i`,true),`<span class="vl">W_xh · x</span>${uvec(xa)}`,'',T(['unit','weights × x','= result'],xa.map((v,j)=>[dot(j),prod(cfg.Wxh[j],x),`<b>${N(v)}</b>`])));}
  if(nm==='mh'){const ha=mv(cfg.Whh,hp);return rCard('Inside the memory weights W_hh · h','W·h',AC.op,'The old memory is mixed by the recurrent weights. Each hidden unit looks at all the old memory numbers.',`<span class="vl">h(t−1)</span>${uvec(hp)}`,K(R`(W_{hh}h)_j=\sum_i W_{hh,ji}\,h_i`,true),`<span class="vl">W_hh · h</span>${uvec(ha)}`,'',T(['unit','weights × h(t−1)','= result'],ha.map((v,j)=>[dot(j),prod(cfg.Whh[j],hp),`<b>${N(v)}</b>`])));}
  if(nm==='sum'){const xa=mv(cfg.Wxh,x),ha=mv(cfg.Whh,hp);return rCard('Inside the adder (+ bias)','+',AC.box,'For every hidden unit the two products and the bias are added. The result s is the raw new memory.',`<span class="vl">W_xh·x</span>${uvec(xa)}<span class="vl">W_hh·h</span>${uvec(ha)}`,K(R`s_j=(W_{xh}x)_j+(W_{hh}h)_j+b_j`,true),`<span class="vl">s(t)</span>${uvec(sv)}`,'',T(['unit','W_xh·x','+ W_hh·h','+ bias','= s'],sv.map((v,j)=>[dot(j),N(xa[j]),N(ha[j]),N(cfg.b[j]),`<b>${N(v)}</b>`])));}
  if(nm==='act'){const o=ht,a=cfg.act;
    if(a==='relu')return rCard('Inside the activation (ReLU)','ReLU','#F97316','ReLU keeps a positive number and turns a negative number into 0.',`<span class="vl">s(t)</span>${uvec(sv)}`,K(R`\mathrm{ReLU}(s)=\max(0,s)`,true),`<span class="vl">h(t)</span>${uvec(o)}`,inCurve('relu',sv,o),T(['unit','s (came in)','max(0, s) = h'],sv.map((v,j)=>[dot(j),N(v),`<b>${N(o[j])}</b>`])));
    if(a==='sigmoid')return rCard('Inside the activation (sigmoid)','σ',AC.sig,'Sigmoid turns any number into a value between 0 and 1.',`<span class="vl">s(t)</span>${uvec(sv)}`,K(R`\sigma(s)=\dfrac{1}{1+e^{-s}}`,true),`<span class="vl">h(t)</span>${uvec(o)}`,inCurve('sig',sv,o),T(['unit','s','e<sup>−s</sup>','1 + e<sup>−s</sup>','1 ÷ that = h'],sv.map((v,j)=>[dot(j),N(v),N(Math.exp(-v)),N(1+Math.exp(-v)),`<b>${N(o[j])}</b>`])));
    return rCard('Inside the activation (tanh)','tanh',AC.tan,'tanh squeezes any number into the range −1 to 1: small numbers stay almost the same, big ones are flattened.',`<span class="vl">s(t)</span>${uvec(sv)}`,K(R`\tanh(s)=\dfrac{e^{s}-e^{-s}}{e^{s}+e^{-s}}`,true),`<span class="vl">h(t)</span>${uvec(o)}`,inCurve('tanh',sv,o),T(['unit','s','e<sup>s</sup>','e<sup>−s</sup>','(e<sup>s</sup>−e<sup>−s</sup>) ÷ (e<sup>s</sup>+e<sup>−s</sup>) = h'],sv.map((v,j)=>[dot(j),N(v),N(Math.exp(v)),N(Math.exp(-v)),`<b>${N(o[j])}</b>`])));}
  if(!oh)return null;
  const u=m.u[t],y=m.y[t],tg=cfg.t[k];
  if(nm==='mo')return rCard('Inside the output layer W_hy · h + b_y','W·h',AC.op,'Each possible answer has its own row of weights that reads the memory and produces a raw score.',`<span class="vl">h(t)</span>${uvec(ht)}`,K(R`u_c=\sum_j W_{hy,cj}h_j+b_{y,c}`,true),`<span class="vl">u(t)</span>${uvec(u)}`,'',T(['output','weights × h','+ b_y','= u'],u.map((v,c)=>[dot(c),prod(cfg.Why[c],ht),N(cfg.by[c]),`<b>${N(v)}</b>`])));
  if(nm==='oact'){
    if(cfg.out==='sigmoid')return rCard('Inside the output function (sigmoid)','σ',AC.sig,'Each score becomes a number between 0 and 1.',`<span class="vl">u(t)</span>${uvec(u)}`,K(R`y=\sigma(u)=\dfrac{1}{1+e^{-u}}`,true),`<span class="vl">y(t)</span>${uvec(y)}`,inCurve('sig',u,y),T(['output','u','e<sup>−u</sup>','1 + e<sup>−u</sup>','1 ÷ that = y'],u.map((v,c)=>[dot(c),N(v),N(Math.exp(-v)),N(1+Math.exp(-v)),`<b>${N(y[c])}</b>`])));
    if(cfg.out==='linear')return rCard('Output (identity)','=',AC.gray,'The scores are used directly as the prediction.',`<span class="vl">u(t)</span>${uvec(u)}`,K(R`y=u`,true),`<span class="vl">y(t)</span>${uvec(y)}`,'',T(['output','u = y'],u.map((v,c)=>[dot(c),`<b>${N(v)}</b>`])));
    const mx=Math.max(...u),ex=u.map(v=>Math.exp(v-mx)),se=ex.reduce((p,q)=>p+q,0);
    return rCard('Inside the output function (softmax)','softmax',AC.sig,'Softmax makes every score positive with e^u, then divides by the total so the answers add up to 1 (100%).',`<span class="vl">u(t)</span>${uvec(u)}`,K(R`y_c=\dfrac{e^{u_c}}{\sum_q e^{u_q}}`,true),`<span class="vl">y(t)</span>${uvec(y)}`,'',T(['output','u','e<sup>u</sup> (shifted by max for safety)','÷ total '+N(se)+' = y'],u.map((v,c)=>[dot(c),N(v),N(ex[c]),`<b>${N(y[c])}</b> (${(y[c]*100).toFixed(1)}%)`])));}
  if(nm==='loss'){const sm=cfg.out==='softmax';
    return rCard('Inside the loss','loss',AC.loss,sm?'Cross-entropy only looks at the probability given to the correct answer: the smaller that is, the bigger the loss.':'The loss is half the sum of the squared differences between prediction and target.',`<span class="vl">y(t)</span>${uvec(y)}<span class="vl">target</span>${uvec(tg)}`,K(sm?R`L=-\sum_c t_c\ln y_c`:R`L=\tfrac12\sum_c(y_c-t_c)^2`,true),`<span class="vl">L(t)</span><span class="uv" style="--uc:${AC.loss}"><i>loss</i>${N(m.lt[t])}</span>`,'',T(sm?['output','target t','y','−t · ln y']:['output','prediction y','target t','½ (y − t)²'],y.map((v,c)=>sm?[dot(c),N(tg[c]),N(v),`<b>${N(-tg[c]*Math.log(v))}</b>`]:[dot(c),N(v),N(tg[c]),`<b>${N(.5*(v-tg[c])**2)}</b>`])));}
  return null;
}
function openInside(nm,t,src){
  const html=rInside(nm,t);if(!html)return;const ov=$('#a-ov'),svg=$('#a-svg');
  ov.innerHTML=`<div class="insidecard"><button class="ix" id="a-ox">✕ Close</button>${html}</div>`;ov.classList.add('on');RA.inside=nm;
  svg.classList.add('zooming');svg.querySelectorAll('.part.focus').forEach(e=>e.classList.remove('focus'));if(src)src.classList.add('focus');
  const card=ov.querySelector('.insidecard');
  if(src){const sr=src.getBoundingClientRect(),cr=card.getBoundingClientRect(),dx=sr.left+sr.width/2-(cr.left+cr.width/2),dy=sr.top+sr.height/2-(cr.top+cr.height/2),sc=Math.max(.06,Math.min(sr.width/cr.width,sr.height/cr.height));
    card.style.transition='none';card.style.transform=`translate(${dx}px,${dy}px) scale(${sc})`;card.style.opacity='0';card.offsetWidth;
    card.style.transition='transform 1.1s cubic-bezier(.2,.8,.2,1),opacity .7s ease';card.style.transform='none';card.style.opacity='1';}
  $('#a-ox').onclick=closeInside;
}
function closeInside(){const ov=$('#a-ov');if(!ov)return;ov.classList.remove('on');ov.innerHTML='';RA.inside=null;const svg=$('#a-svg');if(svg){svg.classList.remove('zooming');svg.querySelectorAll('.part.focus').forEach(e=>e.classList.remove('focus'));}}
/* ---------- guided mode + tab rendering ---------- */
function gStop(){if(RA.g.play){clearInterval(RA.g.play);RA.g.play=null;}}
function gGo(s){RA.g.s=Math.max(0,Math.min(SEQ().length-1,s));renderArch();}
function guidedHTML(){const sq=SEQ(),n=Math.min(RA.g.s,sq.length-1),g=sq[n],t=RA.t;
  const body=RA.g.back?bInfo(g.k,t):g.k.map(k=>rInfo(k,t,true)).join('<hr class="sep">');
  const look=!RA.g.back&&RIN.has(g.k[0])?`<button class="btn lookb" data-look="${g.k[0]}">🔍 Look inside this part</button>`:'';
  return`<div class="gstep"><div class="gk">${RA.g.back?'Backward':'Forward'} · step ${n+1} of ${sq.length} · time step t = ${t+1}</div><h3>${rt(g.t)}</h3>
   <div class="flow"><span class="fb">${esc(g.from)}</span><span class="fa">➜</span><span class="fb to">${esc(g.to)}</span></div>
   <p class="gsay${RA.g.back?' back':''}">${rt(g.say)}</p>${look}${body}</div>`;}
function renderArch(){
  const root=$('#archroot');
  if(!root.dataset.ready){root.dataset.ready='1';
    root.innerHTML=`<div class="archbar"><div class="seg" id="a-run"><button data-r="0">Explore all</button><button data-r="1">▶ Forward, step by step</button><button data-r="2">◀ Backward (BPTT), step by step</button></div>
      <div class="seg" id="a-mode"><button data-m="cell">Inside one cell</button><button data-m="time">Unrolled in time</button></div>
      <div class="seg" id="a-t"></div>
      <button class="btn" id="a-all">Expand all</button><label class="note" id="a-vl"><input type="checkbox" id="a-vals" checked> numbers</label>
      <span class="zoomb"><button class="btn" id="a-zm">−</button><button class="btn" id="a-fit">Fit</button><button class="btn" id="a-zp">+</button></span></div>
      <div class="gbarx" id="a-gbar"><button class="btn" id="g-prev">◀ Back</button><button class="btn pri" id="g-next">Next ▶</button><button class="btn" id="g-play">▶ Play</button><div class="gdots" id="g-dots"></div><button class="btn" id="g-reset">↺ Restart</button></div>
      <div class="archcap" id="a-cap"></div>
      <div class="archgrid"><div class="archstage" id="a-stage"><div class="archwrap" id="a-wrap"><svg id="a-svg"></svg></div><div class="insideov" id="a-ov"></div></div><aside class="archinfo" id="a-info"></aside></div>
      <div class="archleg"><span><i style="background:${AC.x}"></i>Input x</span><span><i style="background:${AC.h}"></i>Memory h (hidden state)</span><span><i style="background:${AC.op}"></i>weights × (matrix product)</span><span><i style="background:${AC.box}"></i>+ add</span><span><i style="background:${AC.tan}"></i>tanh / <i style="background:${AC.sig}"></i>σ activation</span><span><i style="background:${AC.out}"></i>Output y</span><span><i style="background:${AC.loss}"></i>loss</span><span><i style="background:${AC.g}"></i>gradient (backward)</span><span><i style="background:${AC.bias}"></i>bias</span></div>`;
    $('#a-run').onclick=e=>{const b=e.target.closest('button');if(!b)return;gStop();const r=b.dataset.r;RA.g.on=r!=='0';RA.g.back=r==='2';if(RA.g.on){RA.mode='cell';RA.g.s=0;RA.sel=null;if(RA.g.back)RA.t=S.cfg.T-1;}RA.zoom=null;renderArch();};
    $('#a-mode').onclick=e=>{const b=e.target.closest('button');if(!b)return;RA.mode=b.dataset.m;if(RA.mode==='time'){gStop();RA.g.on=false;}RA.sel=null;RA.zoom=null;renderArch();};
    $('#a-t').onclick=e=>{const b=e.target.closest('button');if(!b)return;RA.t=+b.dataset.t;RA.g.s=0;renderArch();};
    $('#a-all').onclick=()=>{const T=S.cfg.T;if(RA.open.size>=T)RA.open.clear();else for(let t=0;t<T;t++)RA.open.add(t);RA.zoom=null;renderArch();};
    $('#a-vals').onchange=e=>{RA.vals=e.target.checked;renderArch();};
    $('#a-zp').onclick=()=>{RA.zoom=(RA.zoom||archFit())*1.2;renderArch();};$('#a-zm').onclick=()=>{RA.zoom=(RA.zoom||archFit())/1.2;renderArch();};$('#a-fit').onclick=()=>{RA.zoom=null;renderArch();};
    $('#g-prev').onclick=()=>{gStop();gGo(RA.g.s-1);};$('#g-next').onclick=()=>{gStop();gGo(RA.g.s+1);};$('#g-reset').onclick=()=>{gStop();gGo(0);};
    $('#g-play').onclick=()=>{if(RA.g.play){gStop();renderArch();return;}if(RA.g.s>=SEQ().length-1)RA.g.s=0;RA.g.play=setInterval(()=>{if(RA.g.s>=SEQ().length-1){gStop();renderArch();return;}gGo(RA.g.s+1);},6500);gGo(RA.g.s);};
    $('#g-dots').onclick=e=>{const d=e.target.closest('i');if(d){gStop();gGo(+d.dataset.i);}};
    document.addEventListener('keydown',e=>{if(tab!=='arch'||!RA.g.on||/INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName))return;if(e.key==='ArrowRight'){gStop();gGo(RA.g.s+1);}if(e.key==='ArrowLeft'){gStop();gGo(RA.g.s-1);}if(e.key==='Escape')closeInside();});
    $('#a-svg').addEventListener('click',e=>{const sb=e.target.closest('[data-act="shrink"]');if(sb){RA.open.delete(+sb.dataset.t);RA.zoom=null;renderArch();return;}
      const g=e.target.closest('.part');if(!g)return;const t=+g.dataset.t;if(g.dataset.act==='expand'){RA.open.add(t);RA.t=t;RA.zoom=null;renderArch();return;}const nm=g.dataset.p;if(GS()&&RA.g.back)return;if(!GS()){RA.t=t;RA.sel=nm;renderArch();}if(RIN.has(nm))openInside(nm,RA.t,document.querySelector(`#a-svg [data-p="${nm}"]`));});
    $('#a-ov').addEventListener('click',e=>{if(e.target.id==='a-ov')closeInside();});
    $('#a-info').addEventListener('click',e=>{const b=e.target.closest('[data-look]');if(b){const nm=b.dataset.look;openInside(nm,RA.t,document.querySelector(`#a-svg [data-p="${nm}"]`));}});
    window.addEventListener('resize',()=>{if(tab==='arch'&&RA.zoom==null)renderArch();});}
  RA.m=forward(S.cfg);RA.gr=backward(S.cfg).g;const T=S.cfg.T;if(RA.t>=T)RA.t=T-1;const gd=GS();
  if(RA.g.s>=SEQ().length)RA.g.s=SEQ().length-1;
  const sg=[RA.g.s,RA.g.on,RA.g.back,RA.mode,RA.t,RA.vals].join();if(RA._sg!==sg){closeInside();RA._sg=sg;}
  document.querySelectorAll('#a-run button').forEach(b=>b.classList.toggle('on',(b.dataset.r==='0'&&!RA.g.on)||(b.dataset.r==='1'&&RA.g.on&&!RA.g.back)||(b.dataset.r==='2'&&RA.g.on&&RA.g.back)));
  document.querySelectorAll('#a-mode button').forEach(b=>b.classList.toggle('on',b.dataset.m===RA.mode));
  $('#a-t').style.display=RA.mode==='cell'?'':'none';$('#a-all').style.display=RA.mode==='time'?'':'none';$('#a-all').textContent=RA.open.size>=T?'Shrink all':'Expand all';
  $('#a-mode').style.display=gd?'none':'';$('#a-vl').style.display=gd?'none':'';$('#a-gbar').style.display=gd?'':'none';
  $('#a-t').innerHTML=[...Array(T).keys()].map(t=>`<button data-t="${t}" class="${t===RA.t?'on':''}">t = ${t+1}</button>`).join('');
  $('#a-cap').style.display=gd?'none':'';$('#a-cap').innerHTML=rt(RA.mode==='time'?`**The same cell, repeated.** All ${T} blocks use exactly the same weights $W_{xh}$, $W_{hh}$, $W_{hy}$ and bias. Only the memory $h$ travels from block to block. Click a block to open it.`:'Every part of one RNN cell. Click any shape to see what it is and its live numbers, or press a step-by-step button.');
  if(gd){const n=SEQ().length;$('#g-dots').innerHTML=SEQ().map((_,i)=>`<i class="${i<RA.g.s?'d':i===RA.g.s?'a':''}" data-i="${i}" title="${esc(SEQ()[i].t)}"></i>`).join('');$('#g-play').textContent=RA.g.play?'⏸ Pause':'▶ Play';$('#g-prev').disabled=RA.g.s===0;$('#g-next').disabled=RA.g.s===n-1;}
  const g=archRSVG(),svg=$('#a-svg'),z=RA.zoom||archFit(g.W,g.H);svg.setAttribute('viewBox',`0 0 ${g.W} ${g.H}`);svg.style.width=Math.round(g.W*z)+'px';svg.style.height=Math.round(g.H*z)+'px';svg.innerHTML=g.html;RA._W=g.W;RA._H=g.H;
  $('#a-info').innerHTML=gd?guidedHTML():(RA.sel?rInfo(RA.sel,RA.t)+(RIN.has(RA.sel)?`<button class="btn lookb" data-look="${RA.sel}">🔍 Look inside this part</button>`:''):`<h3>Click any part</h3><p class="note">Every shape is clickable: the input, the two weight products, the adder, the activation, the memory h, the output layer and the loss. You will see what it is, why it exists and its numbers for the chosen time step.</p><p class="note">Press <b>▶ Forward, step by step</b> to build the cell one operation at a time, or <b>◀ Backward (BPTT)</b> to watch the error flow back. <b>Unrolled in time</b> shows the same cell repeated for every step of the sequence.</p>`);
  root.classList.toggle('guided',gd);
}
function archFit(W,H){W=W||RA._W||960;H=H||RA._H||490;const w=($('#a-wrap')&&$('#a-wrap').clientWidth)||900,hh=Math.max(360,innerHeight-250);return Math.max(.3,Math.min(1.25,(w-8)/W,hh/H));}
