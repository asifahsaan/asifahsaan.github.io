/* ================= words / letters examples, extras (included after the core RNN code) ================= */
const oneHot=(n,i)=>Array.from({length:n},(_,k)=>k===i?1:0);
const tokVec=(cfg,i)=>cfg.tok.emb?cfg.tok.emb[i].slice():oneHot(cfg.tok.vocab.length,i);
const safeTok=s=>String(s).replace(/[^A-Za-z0-9]/g,'');
function tokSync(cfg){const tk=cfg.tok,V=tk.vocab.length;cfg.C=V;cfg.out='softmax';
  const inputs=tk.seq.slice(0,-1);cfg.T=inputs.length;cfg.d=tk.emb?tk.emb[0].length:V;
  cfg.x=inputs.map(i=>tokVec(cfg,i));
  cfg.t=cfg.mode==='all'?tk.seq.slice(1).map(i=>oneHot(V,i)):[oneHot(V,tk.seq[tk.seq.length-1])];return cfg;}
function randWeights(cfg,seed){const r=mulberry(seed),rn=a=>+((r()*2-1)*a).toFixed(2),{d,H,C}=cfg;
  cfg.Wxh=Array.from({length:H},()=>Array.from({length:d},()=>rn(.7)));cfg.Whh=Array.from({length:H},()=>Array.from({length:H},()=>rn(.5)));cfg.b=Array(H).fill(0);
  cfg.Why=Array.from({length:C},()=>Array.from({length:H},()=>rn(.7)));cfg.by=Array(C).fill(0);return cfg;}
const hashStr=s=>[...s].reduce((a,ch)=>(a*31+ch.charCodeAt(0))|0,7)>>>0;
function wordCfg(o){const c=Object.assign(defaultCfg(),{act:'tanh',gran:'neuron',eta:.5,mode:'all',H:3},o);
  c.tok={vocab:o.vocab.slice(),seq:o.seq.map(w=>o.vocab.indexOf(w)),emb:o.emb||null};delete c.vocab;delete c.seq;delete c.emb;
  tokSync(c);if(!o.Wxh)randWeights(c,hashStr(o.vocab.join('')+o.seq.join('')));return c;}
const WORD_PRESETS={
  'Slides example: the word "dogs" (letters d o g s, one-hot, 3 hidden units, next letter at every step)':()=>wordCfg({vocab:['d','o','g','s'],seq:['d','o','g','s'],mode:'all',H:3,
    Wxh:[[.6,.8,.4,.8],[.2,.2,.8,.7],[.9,.8,.1,.2]],Whh:[[.1,.5,.1],[.5,.9,.3],[.3,.2,.1]],b:[0,0,0],Why:[[.9,.8,.3],[.2,.3,.4],[.6,.9,.1],[.5,0,.3]],by:[0,0,0,0]}),
  'Next word: "I love" → AI (7 words, 2-number embeddings, 2 hidden units)':()=>wordCfg({vocab:['I','love','AI','and','learning','is','useful'],seq:['I','love','AI'],mode:'last',H:2,
    emb:[[1,0],[0,1],[1,1],[.5,.5],[.5,.5],[.5,.5],[.5,.5]],Wxh:[[.5,.2],[.1,.4]],Whh:[[.3,.1],[.2,.2]],b:[0,0],Why:[[.1,.1],[.2,.1],[1,1.2],[.3,.2],[.5,.4],[.1,.3],[.4,.5]],by:[0,0,0,0,0,0,0]}),
  'Memory demo: pattern 1 1 0 1 1 (press Train: the hidden state must tell the first 1 from the second)':()=>wordCfg({vocab:['0','1'],seq:['1','1','0','1','1'],mode:'all',H:3,eta:.5})};
/* ---------- tooltips / labels ---------- */
const _cellName=cellName;
cellName=function(key){const cfg=S.cfg;if(cfg.tok){const [p,t,i]=key.split(':');if(p==='x'&&!cfg.tok.emb)return`x(t=${t}) “${cfg.tok.vocab[+i]}” slot`;if(p==='y'||p==='u')return`${p}(t=${t}) for “${cfg.tok.vocab[+i]}”`;}return _cellName(key);};
function updateVocabline(){const el=$('#vocabline'),cfg=S.cfg;if(!cfg.tok){el.innerHTML='';return;}
  const cs=cum(S.i),{m}=backward(cfg),outs=outSteps(cfg),tk=cfg.tok;
  let h=`<b>Vocabulary</b> (position in the input and output rows): `+tk.vocab.map((w,i)=>`<span class="tkn">${i+1}·${esc(w)}</span>`).join('');
  const lines=[];outs.forEach((t,k)=>{if(!cs.vis.has(`y:${t+1}:0`))return;const y=m.y[t],pi=y.indexOf(Math.max(...y)),ti=cfg.t[k].indexOf(1);
    lines.push(`step ${t+1}: input <b>${esc(tk.vocab[tk.seq[t]])}</b> → predicts <b>${esc(tk.vocab[pi])}</b> (${(y[pi]*100).toFixed(1)}%), target <b>${esc(tk.vocab[ti])}</b> gets ${(y[ti]*100).toFixed(1)}%`);});
  el.innerHTML=h+(lines.length?`<div class="vl2">${lines.join('<br>')}</div>`:'');}
const _paint=paint;paint=function(){_paint();updateVocabline();};
$('#learn').addEventListener('toggle',e=>{if(e.target.classList&&e.target.classList.contains('concepts'))window.__conceptsOpen=e.target.open;},true);
/* ---------- training extras ---------- */
function rollout(cfg,tokens,n){const tk=cfg.tok,A=ACTS[cfg.act];let h=Array(cfg.H).fill(0),seq=tokens.slice(),probs=null;
  const step=i=>{const a=mv(cfg.Wxh,tokVec(cfg,i)),r=mv(cfg.Whh,h);h=a.map((v,j)=>A.f(v+r[j]+cfg.b[j]));const u=mv(cfg.Why,h).map((v,c)=>v+cfg.by[c]),mx=Math.max(...u),e=u.map(v=>Math.exp(v-mx)),s=e.reduce((p,q)=>p+q,0);probs=e.map(v=>v/s);return probs;};
  for(const i of tokens)step(i);const gen=[];for(let q=0;q<n;q++){const pi=probs.indexOf(Math.max(...probs));gen.push([pi,probs[pi]]);step(pi);}return{gen};}
const hv=(v)=>`(${v.map(x=>x.toFixed(3)).join(', ')})`;
function trainExtra(){const el=$('#textra'),cfg=S.cfg;if(!TR.cfg||!TR.start){el.innerHTML='';return;}
  const m0=forward(TR.start),m1=forward(TR.cfg),outs=outSteps(cfg),tk=cfg.tok;
  const inTok=t=>tk?esc(tk.vocab[tk.seq[t]]):`x${t+1}`;
  let h=`<h3 style="font-size:15px;margin:18px 0 6px">Hidden state after each step: start → now</h3><div class="note">Different inputs, or the same input at a different position, should leave different hidden states if the network has learned to remember.</div><table class="t"><tr><th>step</th><th>input</th><th>h at start</th><th>h now</th></tr>`
    +Array.from({length:cfg.T},(_,t)=>`<tr><td>${t+1}</td><td>${inTok(t)}</td><td>${hv(m0.h[t+1])}</td><td><b>${hv(m1.h[t+1])}</b></td></tr>`).join('')+'</table>';
  if(cfg.out==='softmax'){const nm=i=>tk?esc(tk.vocab[i]):`class ${i+1}`;
    const row=(t,k)=>{const f=m=>{const y=m.y[t],pi=y.indexOf(Math.max(...y)),ti=cfg.t[k].indexOf(1);return{pi,ti,py:y[pi],pt:y[ti]};},a=f(m0),b=f(m1);return`<tr><td>${t+1}</td><td>${inTok(t)}</td><td>${nm(a.ti)}</td><td>${nm(a.pi)} (${(a.py*100).toFixed(1)}%) · target ${(a.pt*100).toFixed(1)}%</td><td><b>${nm(b.pi)} (${(b.py*100).toFixed(1)}%)</b> · target ${(b.pt*100).toFixed(1)}%</td></tr>`;};
    h+=`<h3 style="font-size:15px;margin:18px 0 6px">Predictions: untrained → trained</h3><table class="t"><tr><th>step</th><th>input</th><th>target</th><th>at start</th><th>now</th></tr>${outs.map((t,k)=>row(t,k)).join('')}</table>`;}
  if(tk){const toks=tk.seq.slice(0,cfg.T),a=rollout(TR.start,toks,5).gen,b=rollout(TR.cfg,toks,5).gen,nm=i=>esc(tk.vocab[i]);
    h+=`<h3 style="font-size:15px;margin:18px 0 6px">Generate text: feed each prediction back as the next input</h3><div class="note">Start with “${toks.map(i=>esc(tk.vocab[i])).join(' ')}” and always take the most likely next token (greedy). This is exactly how a language model writes text, one token at a time.</div>
      <table class="t"><tr><th></th><th style="text-align:left">generated continuation</th></tr><tr><td>start</td><td style="text-align:left">${toks.map(nm).join(' ')} <span style="color:#7C4DDB">→ ${a.map(([i,p])=>nm(i)).join(' ')}</span></td></tr><tr><td>now</td><td style="text-align:left">${toks.map(nm).join(' ')} <b style="color:#7C4DDB">→ ${b.map(([i,p])=>nm(i)).join(' ')}</b></td></tr></table>`;}
  el.innerHTML=h;}
const _renderTrain=renderTrain;renderTrain=function(){_renderTrain();const cfg=S.cfg;
  if(cfg.tok){const m=forward(TR.cfg),outs=outSteps(cfg);$('#tbars').innerHTML=outs.map((t,k)=>`<div style="font-size:12px;margin:6px 0 2px;color:#6b7280">step ${t+1}: input “${esc(cfg.tok.vocab[cfg.tok.seq[t]])}”, next token should be “${esc(cfg.tok.vocab[cfg.t[k].indexOf(1)])}”</div>`+m.y[t].map((v,c)=>`<div class="b"><span style="width:54px;font-size:12px">${esc(cfg.tok.vocab[c])}</span><div class="track"><div class="fill" style="width:${Math.max(0,Math.min(100,v*100)).toFixed(1)}%"></div>${cfg.t[k][c]===1?`<div class="tg" style="left:98.5%"></div>`:''}</div><span style="width:70px;text-align:right">${v.toFixed(4)}</span></div>`).join('')).join('');}
  trainExtra();};
/* ---------- gradient check extras: influence of every input ---------- */
const _renderCheck=renderCheck;renderCheck=function(){_renderCheck();const cfg=S.cfg,{g}=backward(cfg),nrm=v=>Math.sqrt(v.reduce((s,x)=>s+x*x,0));
  const dx=g.ds.map(ds=>cfg.Wxh[0].map((_,i)=>cfg.Wxh.reduce((s,row,j)=>s+row[i]*ds[j],0))),vals=dx.map(nrm),mx=Math.max(1e-12,...vals);
  $('#influence').innerHTML=vals.map((v,t)=>`<div class="b"><span style="width:150px">${cfg.tok?`input ${t+1}: “${esc(cfg.tok.vocab[cfg.tok.seq[t]])}”`:`input $x_${t+1}$`}</span><div class="tr"><div class="fl" style="width:${(v/mx*100).toFixed(1)}%;background:#2B7A9B"></div></div><span style="width:92px;text-align:right">${v.toExponential(2)}</span></div>`).join('');
  document.querySelectorAll('#influence span').forEach(el=>{if(el.textContent.includes('$'))el.innerHTML=rt(el.textContent);});};
/* ---------- custom editor: words / letters ---------- */
function setMsg(t,bad){$('#cmsg').innerHTML=bad?`<span class="warn">${esc(t)}</span>`:`<span class="okk">${esc(t)}</span>`;}
function windowsInfo(sq){const N=sq.length;if(N<2)return'';
  const rows=[];for(let L=1;L<=Math.min(N-1,4);L++){const ex=[];for(let i=0;i+L<N;i++)ex.push(`${sq.slice(i,i+L).join(' ')} → ${sq[i+L]}`);rows.push(`<tr><td>${L}</td><td><b>${N-L}</b></td><td style="text-align:left">${ex.map(esc).join(' &nbsp;·&nbsp; ')}</td></tr>`);}
  return`<div class="fbox" style="max-width:760px"><div class="k">How many training sequences does this text give?</div><div class="note" style="margin-bottom:6px">The system slides a window over the text. A text of <b>N = ${N}</b> tokens and window length <b>L</b> gives <b>N − L</b> examples (L inputs, the next token is the answer). This lab shows <b>one</b> example at a time (the sequence you typed); real training uses all of them.</div><table class="t"><tr><th>window L</th><th>examples</th><th style="text-align:left">the examples</th></tr>${rows.join('')}</table></div>`;}
function buildWords(){const cfg=S.cfg,tk=cfg.tok,el=$('#c-words');
  const vocab=tk?tk.vocab.join(' '):'d o g s',seq=tk?tk.seq.map(i=>tk.vocab[i]).join(' '):'d o g',emb=tk&&tk.emb?String(tk.emb[0].length):'onehot',mode=tk?cfg.mode:'all',H=tk?cfg.H:3;
  el.innerHTML=`<div class="fbox"><div class="k">Your own words or letters</div><div class="row">
    <label>Vocabulary (space separated)<input type="text" id="wv" value="${esc(vocab)}" style="width:300px"></label>
    <label>Sequence (space separated, last token = the answer)<input type="text" id="ws" value="${esc(seq)}" style="width:260px"></label></div><div class="row">
    <label>The network predicts<select id="wm"><option value="all" ${mode==='all'?'selected':''}>the next token at every step</option><option value="last" ${mode==='last'?'selected':''}>only the token after the last input</option></select></label>
    <label>Numbers per word<select id="we"><option value="onehot" ${emb==='onehot'?'selected':''}>one-hot (one slot per vocabulary word)</option><option value="2" ${emb==='2'?'selected':''}>2 numbers (embedding)</option><option value="3" ${emb==='3'?'selected':''}>3 numbers (embedding)</option></select></label>
    <label>Hidden units<input type="number" id="wh" min="1" max="5" value="${H}"></label>
    <button class="btn pri" id="wbuild">Build this example</button></div>
    <div class="note">Example: vocabulary “d o g s”, sequence “d o g” gives inputs d, o and targets o, g (the slides' example). Sequence “I love AI” with “only the token after the last input” asks: after “I love”, what comes next? (2–8 words, letters or digits only, sequence of 2–6 tokens.)</div></div>
    <div id="wemb"></div><div id="wwin"></div>`;
  const win=()=>{const sq=$('#ws').value.trim().split(/[\s,]+/).filter(Boolean);$('#wwin').innerHTML=windowsInfo(sq);};win();$('#ws').oninput=win;
  $('#wbuild').onclick=()=>{const tokens=s=>s.trim().split(/[\s,]+/).filter(Boolean);const vocab=tokens($('#wv').value),seq=tokens($('#ws').value),modeV=$('#wm').value,embV=$('#we').value,Hn=parseInt($('#wh').value);
    if(vocab.length<2||vocab.length>8)return setMsg('Vocabulary needs 2 to 8 different words.',1);
    if(vocab.some(w=>!/^[A-Za-z0-9'-]+$/.test(w)))return setMsg('Use only letters, digits, apostrophes or hyphens in words.',1);
    if(new Set(vocab).size!==vocab.length)return setMsg('Every word in the vocabulary must be different.',1);
    if(seq.length<2||seq.length>6)return setMsg('The sequence needs 2 to 6 tokens (the last one is the answer).',1);
    const bad=seq.find(w=>!vocab.includes(w));if(bad)return setMsg(`“${bad}” is not in the vocabulary.`,1);
    if(!(Hn>=1&&Hn<=5))return setMsg('Hidden units: 1 to 5.',1);
    const old=S.cfg.tok&&S.cfg.tok.emb&&S.cfg.tok.vocab.length===vocab.length&&String(S.cfg.tok.emb[0].length)===embV?S.cfg.tok.emb:null;
    const r=mulberry(hashStr(vocab.join('')+'e'));const emb=embV==='onehot'?null:(old||vocab.map(()=>Array.from({length:+embV},()=>+((r()*2-1)).toFixed(1))));
    const c=wordCfg({vocab,seq,mode:modeV,emb,H:Hn,eta:S.cfg.eta||.5,act:S.cfg.act||'tanh',gran:S.cfg.gran});
    applyCfg(c,`Built: ${vocab.length} words, ${c.T} input step${c.T>1?'s':''}, ${Hn} hidden units. Weights are random: edit them below, or press Train to see the network learn.`);buildCustom();};
  const em=$('#wemb');if(tk&&tk.emb)em.innerHTML=`<div class="fbox"><div class="k">Embedding table: numbers for every word</div><div class="mgrid" style="grid-template-columns:auto repeat(${tk.emb[0].length},auto)">${tk.vocab.map((w,i)=>`<b style="align-self:center;padding-right:6px">${esc(w)}</b>`+tk.emb[i].map((v,j)=>`<input type="number" step="0.1" data-emb="${i},${j}" value="${v}">`).join('')).join('')}</div></div>`;else em.innerHTML='';
  em.querySelectorAll('input[data-emb]').forEach(inp=>inp.oninput=()=>{const v=parseFloat(inp.value);if(!isFinite(v))return;const [i,j]=inp.dataset.emb.split(',').map(Number);S.cfg.tok.emb[i][j]=v;tokSync(S.cfg);TR.cfg=null;rebuild(false);});}
const _buildCustom=buildCustom;buildCustom=function(){_buildCustom();buildWords();const cfg=S.cfg;
  if(cfg.tok){['cT','cd','cH','cC','cout','cmode'].forEach(id=>{const e=$('#'+id);if(e)e.disabled=true;});
    const rows=cfg.x.map((r,t)=>`<tr><td>${t+1}</td><td><b>${esc(cfg.tok.vocab[cfg.tok.seq[t]])}</b></td><td>${hv(r)}</td></tr>`).join('');
    const trows=cfg.t.map((r,k)=>{const t=cfg.mode==='all'?k:cfg.T-1;return`<tr><td>${t+1}</td><td><b>${esc(cfg.tok.vocab[r.indexOf(1)])}</b></td><td>${hv(r)}</td></tr>`;}).join('');
    $('#c-io').innerHTML=`<div class="note" style="margin-bottom:6px">Inputs and targets come from your words. Sizes and the softmax output are fixed by the vocabulary. <button class="btn" id="tonum">Switch to numeric mode</button></div><div class="fbox"><div class="k">Inputs x</div><table class="t"><tr><th>step</th><th>token</th><th>numbers</th></tr>${rows}</table></div><div class="fbox"><div class="k">Targets (next token)</div><table class="t"><tr><th>after step</th><th>token</th><th>one-hot</th></tr>${trows}</table></div>`;
    $('#tonum').onclick=()=>{const c=clone(S.cfg);delete c.tok;applyCfg(c,'Numeric mode: x and t are now plain numbers you can edit.');buildCustom();};}};
