/* ================= Architecture tab: our own modern LSTM cell, clickable + guided ================= */
const AC={belt:'#0891B2',h:'#F59E0B',x:'#475569',out:'#7C3AED',sig:'#F5B50A',tan:'#10B981',op:'#3B82F6',box:'#E8EEF6',bias:'#F472B6',edge:'#334155'};
const A={mode:'cell',t:0,open:new Set(),sel:null,focus:'all',zoom:null,vals:true,peep:false,m:null,g:{on:false,s:0,play:null}};
const FOC={forget:'f',input:'i',cell:'u',output:'o'};
const FCAP={all:'Every part of the cell. Click any shape to see what it is and its live numbers.',
 forget:'**Forget gate.** Controls what information to throw away from memory: $f_t=\\sigma(W_f[h_{t-1},x_t]+b_f)$.',
 input:'**Input gate + candidate.** How much of the new information is written: $i_t=\\sigma(W_i[h_{t-1},x_t]+b_i)$ and $\\tilde c_t=\\tanh(W_c[h_{t-1},x_t]+b_c)$.',
 cell:'**Memory (cell state) update.** The cell state forgets some information, then is updated by adding new information: $c_t=f_t\\odot c_{t-1}+i_t\\odot\\tilde c_t$.',
 output:'**Output gate + hidden state.** A filtered version of the cell state becomes the new output: $o_t=\\sigma(W_o[h_{t-1},x_t]+b_o)$, $h_t=o_t\\odot\\tanh(c_t)$.'};
const MK={h:AC.h,x:AC.x,c:AC.belt,g:AC.belt,e:AC.edge,o:AC.out};
const CXG={f:190,i:330,g:470,o:560},GI=['f','i','g','o'];
const vs=v=>'['+v.slice(0,3).map(f2).join(', ')+(v.length>3?', …':'')+']';
const GS=()=>A.g.on&&A.mode==='cell';
const dm=tags=>(GS()||A.focus==='all'||tags.split(' ').includes(FOC[A.focus]))?'':'dim';
const V=rv=>GS()&&rv!=null?(rv>A.g.s?'hid':rv===A.g.s?'enter':''):'';
const cls=(tags,rv)=>(dm(tags)+' '+V(rv)).trim();
const Ln=(pts,mk,tags,rv,arrow=true,w=3)=>{const d=`M${pts.map(p=>p.join(' ')).join(' L')}`,c=cls(tags,rv);
  return `<path class="ln ${c}" pathLength="1" d="${d}" fill="none" stroke="${MK[mk]}" stroke-width="${w}" stroke-linecap="round" stroke-linejoin="round"${arrow?` marker-end="url(#ah-${mk})"`:''}/>`+(V(rv)==='enter'?`<path class="flowdash" pathLength="1" d="${d}" fill="none" stroke-width="${w}" stroke-linecap="round" stroke-linejoin="round"/>`:'');};
const lab=(b,s)=>`${b}<tspan dy="3" font-size="9">${s}</tspan>`;
const circ=(x,y,r,fill,tags,rv,txt,col='#fff',fs=13)=>`<g class="nd ${cls(tags,rv)}"><circle cx="${x}" cy="${y}" r="${r}" fill="${fill}" stroke="#fff" stroke-width="3"/><text x="${x}" y="${y+4}" text-anchor="middle" font-size="${fs}" font-weight="700" fill="${col}">${txt}</text></g>`;
const chip=(t,x,y,anc,rv)=>{if(!A.vals)return'';const w=t.length*6.3+12,x0=anc==='start'?x:anc==='end'?x-w:x-w/2;return`<g class="${V(rv)}"><rect class="chipbg" x="${x0}" y="${y-13}" width="${w}" height="19" rx="9.5"/><text class="chip" x="${x0+w/2}" y="${y}" text-anchor="middle">${t}</text></g>`;};
const tag=(t,x,y,anc,rv)=>`<text class="ntag ${V(rv)}" x="${x}" y="${y}" text-anchor="${anc||'middle'}">${t}</text>`;
const PT=(nm,t,title,inner)=>`<g class="part${A.sel===nm&&A.t===t&&!GS()?' sel':''}" data-p="${nm}" data-t="${t}"><title>${title}</title>${inner}</g>`;
const sigNode=(x,y,tags,rv)=>`<g class="nd ${cls(tags,rv)}"><circle cx="${x}" cy="${y}" r="23" fill="${AC.sig}" stroke="#fff" stroke-width="3"/><text x="${x}" y="${y+8}" text-anchor="middle" font-size="25" font-family="Georgia,serif" font-style="italic" fill="#3b2f00">σ</text></g>`;
const tanNode=(x,y,tags,rv)=>`<g class="nd ${cls(tags,rv)}"><rect x="${x-25}" y="${y-21}" width="50" height="42" rx="14" fill="${AC.tan}" stroke="#fff" stroke-width="3"/><text x="${x}" y="${y+5}" text-anchor="middle" font-size="13" font-weight="700" fill="#fff">tanh</text></g>`;
const mulNode=(x,y,tags,rv)=>`<g class="nd ${cls(tags,rv)}"><circle cx="${x}" cy="${y}" r="14" fill="${AC.op}" stroke="#fff" stroke-width="3"/><text x="${x}" y="${y+6}" text-anchor="middle" font-size="20" font-weight="700" fill="#fff">×</text></g>`;
const addBox=(x,y,w,tags,rv,sym='+')=>`<g class="nd ${cls(tags,rv)}"><rect x="${x-w/2}" y="${y-14}" width="${w}" height="28" rx="9" fill="${AC.box}" stroke="#A9BBD0" stroke-width="1.5"/><text x="${x}" y="${y+7}" text-anchor="middle" font-size="20" font-weight="700" fill="#3C5068">${sym}</text></g>`;
const GRV={f:{cat:1,sg:2},i:{cat:4,sg:5},g:{cat:6,sg:7},o:{cat:10,sg:11}};

function expCell(t,ox,left,right,showShrink){
  const cfg=S.cfg,M=A.m,G=M.G[t],xv=cfg.x[t],hp=M.h[t],cp=M.c[t],ct=M.c[t+1],ht=M.h[t+1],tt=t+1,gd=GS();let s=`<g transform="translate(${ox},0)">`;
  s+=`<rect class="cellbox" x="100" y="40" width="700" height="360" rx="34"/>`;
  s+=`<rect class="zone zf" x="140" y="96" width="102" height="224" rx="18"/><rect class="zone zi" x="282" y="96" width="240" height="224" rx="18"/><rect class="zone zo" x="534" y="96" width="198" height="224" rx="18"/>`;
  s+=`<text class="zl" x="191" y="112" text-anchor="middle">FORGET</text><text class="zl" x="402" y="112" text-anchor="middle">INPUT</text><text class="zl" x="633" y="112" text-anchor="middle">OUTPUT</text>`;
  s+=`<text class="cap" x="120" y="388">LSTM cell · t = ${tt}</text>`;
  if(showShrink)s+=`<g class="shrinkbtn" data-act="shrink" data-t="${t}"><rect x="690" y="368" width="94" height="22" rx="11"/><text x="737" y="383" text-anchor="middle">– shrink</text></g>`;
  if(A.peep&&!gd)s+=`<g class="peep">`+Ln([[120,70],[120,262],[583,262]],'c','peep',null,false,2)+GI.map(k=>Ln([[CXG[k]-20,262],[CXG[k]-20,277]],'c','peep',null,true,2)).join('')+`<text x="126" y="256" font-size="10" fill="#4a7b8c">peephole links (not used in this lab's maths)</text></g>`;
  /* inputs */
  s+=PT('x',t,'x(t): current input',circ(330,440,19,AC.x,'f i u o',0,lab('x','t'))+chip(vs(xv),360,444,'start',0));
  s+=PT('hprev',t,'h(t-1): previous output',circ(45,345,17,AC.h,'f i u o',0,lab('h','t-1'))+chip(vs(hp),45,378,'middle',0));
  s+=PT('cprev',t,'c(t-1): memory from the previous step',circ(45,70,17,AC.belt,'f u',0,lab('c','t-1'))+chip(vs(cp),45,42,'middle',0));
  /* four mixers with their wires and biases */
  GI.forEach((k,n)=>{const tg=k==='f'?'f u':k==='o'?'o':'i u',cx=CXG[k],r=GRV[k].cat;
    s+=PT('cat'+n,t,`W·[h(t-1), x(t)] for the ${GN[k]}`,
      Ln([[62,345],[cx-24,345],[cx-24,303]],'h',tg,r)+Ln([[330,421],[330,372],[cx,372],[cx,303]],'x',tg,r)+addBox(cx,290,66,tg,r));
    s+=PT('b'+n,t,`bias b of the ${GN[k]}`,Ln([[cx+24,312],[cx+24,303]],'e',tg,r,true,2)+circ(cx+24,323,10,AC.bias,tg,r,n,'#fff',11));});
  /* memory belt */
  s+=PT('cline',t,'cell state: the conveyor belt',Ln([[62,70],[175,70]],'c','f u',3)+Ln([[205,70],[441,70]],'g','u',9)+Ln([[499,70],[right?830:880,70]],'g','u o',9));
  s+=PT('mulf',t,'forget: f ⊙ c(t-1)',mulNode(190,70,'f u',3)+tag('Forget Guard',190,50,'middle',3));
  s+=PT('addc',t,'add new memory: + i ⊙ c̃',addBox(470,70,54,'u',9)+tag('Basket',470,50,'middle',9)+chip('c = '+vs(ct),504,98,'start',9));
  /* forget path */
  s+=PT('sf',t,'forget gate σ',Ln([[190,277],[190,240]],'e','f u',2)+sigNode(190,215,'f u',2)+Ln([[190,191],[190,86]],'e','f u',3)+chip('f = '+vs(G.f),198,180,'start',2));
  /* input path */
  s+=PT('si',t,'input gate σ',Ln([[330,277],[330,240]],'e','i u',5)+sigNode(330,215,'i u',5)+Ln([[330,191],[330,140],[454,140]],'e','i u',8)+chip('i = '+vs(G.i),338,180,'start',5));
  s+=PT('tg',t,'candidate tanh',Ln([[470,277],[470,238]],'e','i u',7)+tanNode(470,215,'i u',7)+Ln([[470,193],[470,156]],'e','i u',8)+chip('c̃ = '+vs(G.g),478,180,'start',7));
  s+=PT('muli',t,'i ⊙ c̃',mulNode(470,140,'i u',8)+tag('Scribe',490,136,'start',8)+Ln([[470,126],[470,86]],'e','i u',9));
  /* output path */
  s+=PT('so',t,'output gate σ',Ln([[560,277],[560,240]],'e','o',11)+sigNode(560,215,'o',11)+Ln([[583,215],[664,215]],'e','o',13)+chip('o = '+vs(G.o),568,262,'start',11));
  s+=PT('tanhc',t,'tanh(c(t))',Ln([[680,70],[680,118]],'g','o',12,false)+tanNode(680,140,'o',12)+Ln([[680,162],[680,200]],'e','o',12)+chip('tanh(c)',652,146,'end',12));
  s+=PT('mulh',t,'h = o ⊙ tanh(c)',mulNode(680,215,'o',13)+tag('Speaker',698,232,'start',13));
  s+=PT('hup',t,'h(t) goes up to the output layer',Ln([[680,229],[680,345],[740,345],[740,44]],'o','o',14)+circ(740,25,17,AC.out,'o',14,lab('h','t'))+chip(vs(ht),772,28,'start',14));
  s+=PT('hout',t,'h(t): output to the next step',(right?Ln([[740,345],[830,345]],'o','o',14)+circ(850,345,17,AC.out,'o',14,lab('h','t'))+chip(vs(ht),850,378,'middle',14):Ln([[740,345],[880,345]],'o','o',14)));
  s+=PT('ct',t,'c(t): new memory',right?circ(850,70,17,AC.belt,'u o',14,lab('c','t'))+chip(vs(ct),850,42,'middle',14):'');
  return s+'</g>';
}
function cmpCell(t,ox,left,right){
  const M=A.m,ct=M.c[t+1],ht=M.h[t+1],tt=t+1,cp=M.c[t],hp=M.h[t],xv=S.cfg.x[t];let s=`<g transform="translate(${ox},0)">`;
  s+=`<g class="part cmp${A.t===t?' sel':''}" data-act="expand" data-t="${t}"><title>Click to expand this cell</title><rect class="cellbox" x="50" y="40" width="140" height="360" rx="26"/>`+
   `<text x="120" y="190" text-anchor="middle" font-size="24" font-weight="700" fill="#33414d">LSTM</text><text x="120" y="218" text-anchor="middle" font-size="16" fill="#51606c">t = ${tt}</text>`+
   `<text x="120" y="262" text-anchor="middle" font-size="12" fill="#1E5873">▸ click to expand</text></g>`;
  s+=Ln([[left?36:0,70],[50,70]],'c','f u',null,false,3)+Ln([[190,70],[right?204:240,70]],'g','u o',null,true,3)+(left?circ(20,70,16,AC.belt,'f u',null,lab('c','t-1'))+chip(vs(cp),20,42,'middle'):'');
  s+=Ln([[left?36:0,345],[50,345]],'h','f i u o',null,true,3)+Ln([[190,345],[right?204:240,345]],'o','o',null,true,3)+(left?circ(20,345,16,AC.h,'f i u o',null,lab('h','t-1'))+chip(vs(hp),20,378,'middle'):'');
  s+=Ln([[120,422],[120,400]],'x','f i u o',null,true,3)+circ(120,440,18,AC.x,'f i u o',null,lab('x','t'))+chip(vs(xv),146,444,'start');
  s+=Ln([[150,40],[150,43]],'o','o',null,false,3)+circ(150,25,16,AC.out,'o',null,lab('h','t'))+chip(vs(ht),176,28,'start');
  if(right)s+=circ(220,70,16,AC.belt,'u o',null,lab('c','t'))+chip(vs(ct),220,42,'middle')+circ(220,345,16,AC.out,'o',null,lab('h','t'))+chip(vs(ht),220,378,'middle');
  else s+=chip('c'+tt+' = '+vs(ct),230,56,'end');
  return s+'</g>';
}
function archSVG(){
  const cfg=S.cfg,T=cfg.T,times=A.mode==='cell'?[A.t]:[...Array(T).keys()];let off=30,body='';
  times.forEach((t,n)=>{const exp=A.mode==='cell'||A.open.has(t),left=n===0,right=n===times.length-1,w=exp?880:240;
    body+=exp?expCell(t,off,left,right,A.mode==='time'):cmpCell(t,off,left,right);off+=w;});
  const W=off+50,Hh=470;
  const defs=`<defs>${Object.entries(MK).map(([k,c])=>`<marker id="ah-${k}" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto"><path d="M0,0L10,5L0,10z" fill="${c}"/></marker>`).join('')}</defs>`;
  return{W,H:Hh,html:`${defs}${body}`};
}
function archInfo(nm,t,noHead){
  const cfg=S.cfg,M=A.m,P=cfg.p,x=cfg.x[t],hp=M.h[t],cp=M.c[t],G=M.G[t],ct=M.c[t+1],ht=M.h[t+1],tt=t+1,th=ct.map(Math.tanh);
  const had=(a,b)=>a.map((v,j)=>v*b[j]);const D=s=>K(s,true);
  const gk=nm.startsWith('cat')||nm.startsWith('b')&&nm.length===2?GI[+nm.slice(nm.startsWith('cat')?3:1)]:nm==='sf'?'f':nm==='si'?'i':nm==='tg'?'g':nm==='so'?'o':null;
  const pct=v=>Math.round(v*100)+'%';
  let title='',what='',tex=[],read='',why='';
  if(nm==='cprev'){title=R`Cell state from the previous step`;what=`$c_{t-1}$ is the **memory** that arrives from step ${t}. It is the conveyor belt of the LSTM: it runs straight through the cell and is only changed by two small operations (forget and add).`;tex=[R`c_{${t}}=${bmc(cp)}`];if(t===0)read='At the first step the memory starts as zeros.';why='A separate memory lane lets information travel many steps with almost no change, which is what keeps gradients alive.';}
  else if(nm==='hprev'){title='Hidden state from the previous step';what=`$h_{t-1}$ is the **output** of the previous step. It is shown to every gate so that each gate knows what was just produced.`;tex=[R`h_{${t}}=${bmc(hp)}`];if(t===0)read='At the first step it starts as zeros.';}
  else if(nm==='x'){title='Current input';what=`$x_t$ is the new information at time step ${tt}. It goes to all four combiners together with $h_{t-1}$.`;tex=[R`x_{${tt}}=${bmc(x)}`];}
  else if(nm.startsWith('cat')){const k=gk,w=WS[k];title=`Combiner for the ${GN[k]}`;what=`The **+** box combines $x_t$ and $h_{t-1}$ with the gate's own weights, then adds its bias. In the slides this is $W[h_{t-1},x_t]+b$. Here the big weight matrix is stored as two blocks, $W$ (for $x$) and $U$ (for $h$), which gives the same result.`;
    const z=preAct(P,k,x,hp);tex=[R`z_${w}=W_${w}x_t+U_${w}h_{t-1}+b_${w}`,R`=${bmM(P[k].W)}${bmc(x)}+${bmM(P[k].U)}${bmc(hp)}+${bmc(P[k].b)}=${bmc(z)}`];why='Each gate has its own weights, so each gate can learn to look at the input and the old output in a different way.';}
  else if(nm.length===2&&nm[0]==='b'){const k=gk,w=WS[k];title=`Bias of the ${GN[k]}`;what=`A bias $b_${w}$ is a number added before the activation. It shifts the gate: a large positive bias keeps a gate open by default.`;tex=[R`b_${w}=${bmc(P[k].b)}`];}
  else if(gk&&nm.length===2&&nm[0]==='s'||nm==='tg'){const k=gk,z=preAct(P,k,x,hp);
    if(k==='g'){title='Candidate (tanh)';what=`The **tanh** layer proposes new content $\\tilde c_t$ to remember. tanh squeezes every number into the range −1 to 1.`;tex=[R`\tilde c_t=\tanh(z_c)=\tanh${bmc(z)}=${bmc(G.g)}`];why='New content can be positive or negative, so the cell state can be increased or decreased.';}
    else{title=`${GN[k][0].toUpperCase()+GN[k].slice(1)} (sigmoid)`;what=`A **sigmoid** squeezes numbers to between 0 and 1. A value of 0 means "let nothing through", 1 means "let everything through". This gate ${GM[k]}`;
      tex=[R`${k}_t=\sigma(z_${WS[k]})=\sigma${bmc(z)}=${bmc(G[k])}`];
      read=G[k].map((v,j)=>`unit ${j+1}: ${k==='f'?`keeps ${pct(v)} of the old memory`:k==='i'?`writes ${pct(v)} of the candidate`:`reveals ${pct(v)} of the memory`}`).join(' · ');}}
  else if(nm==='mulf'){title='Forget: multiply the old memory';const pr=had(G.f,cp);what='Element-wise product: each memory cell is multiplied by its forget value. 0 erases it, 1 keeps it.';tex=[R`f_t\odot c_{t-1}=${bmc(G.f)}\odot${bmc(cp)}=${bmc(pr)}`];}
  else if(nm==='muli'){title='Write: multiply candidate by the input gate';const pr=had(G.i,G.g);what='The input gate decides how much of the candidate really gets written.';tex=[R`i_t\odot\tilde c_t=${bmc(G.i)}\odot${bmc(G.g)}=${bmc(pr)}`];}
  else if(nm==='addc'||nm==='cline'||nm==='ct'){title=nm==='ct'?'New cell state':'Memory update (add)';const a=had(G.f,cp),b=had(G.i,G.g);what='The kept memory and the newly written content are **added**. This one plus is the main trick of the LSTM: addition lets gradients flow back without shrinking.';tex=[R`c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t`,R`=${bmc(a)}+${bmc(b)}=${bmc(ct)}`];}
  else if(nm==='tanhc'){title='tanh of the cell state';what='The cell state can grow large, so it is squashed with tanh to the range −1 to 1 before it is revealed.';tex=[R`\tanh(c_t)=\tanh${bmc(ct)}=${bmc(th)}`];}
  else if(nm==='mulh'||nm==='hout'||nm==='hup'){title='Hidden state (output of this step)';what=nm==='hup'?'The same $h_t$ also goes **up** to the output layer, where the prediction $y_t$ is computed.':'The output gate filters the squashed memory. The result $h_t$ is sent to the next step and to the output layer.';tex=[R`h_t=o_t\odot\tanh(c_t)=${bmc(G.o)}\odot${bmc(th)}=${bmc(ht)}`];}
  else{title='Cell';what='Click a shape in the diagram.';}
  return`${noHead?`<h4 class="ih">${rt(title)}</h4>`:`<h3>${rt(title)} <span class="tbadge">t = ${tt}</span></h3>`}<p>${rt(what)}</p>${tex.map(D).join('')}${read?`<div class="read">${esc(read)}</div>`:''}${why?`<p class="why"><b>Why it exists:</b> ${rt(why)}</p>`:''}`;
}
const GSTEPS=[
 {t:'The three inputs arrive',from:'the previous step and the new word',to:'the LSTM cell',k:['x','hprev','cprev'],say:'**x** is the new word. **h(t−1)** is what the cell said last time. **c(t−1)** is its long-term memory. Nothing has happened yet.'},
 {t:'Forget gate: the Mixer',from:'x and h(t−1)',to:'Mixer 0',k:['cat0'],say:'The Mixer blends the new word with the last output, using the forget gate\'s own weights and bias.'},
 {t:'Forget gate: the Dimmer switch',from:'Mixer 0',to:'Dimmer σ',k:['sf'],say:'The Dimmer turns the mixed number into a dial **f** between 0 and 1.'},
 {t:'The Forget Guard meets the old memory',from:'c(t−1) and dial f',to:'kept memory',k:['mulf'],say:'The guard multiplies the old memory by f. If f is near 0 the memory is wiped, near 1 it passes.'},
 {t:'Input gate: the Mixer',from:'x and h(t−1)',to:'Mixer 1',k:['cat1'],say:'A second Mixer, with different weights, prepares the input gate.'},
 {t:'Input gate: the Dimmer switch',from:'Mixer 1',to:'Dimmer σ',k:['si'],say:'Dial **i** (0 to 1) says how much of the new idea may be written.'},
 {t:'Candidate: the Mixer',from:'x and h(t−1)',to:'Mixer 2',k:['cat2'],say:'A third Mixer prepares the new content.'},
 {t:'Candidate: the Dreamer',from:'Mixer 2',to:'Dreamer (tanh)',k:['tg'],say:'The Dreamer proposes new content **c̃** between −1 and 1.'},
 {t:'The Scribe writes the dream',from:'dial i and the dream c̃',to:'written content',k:['muli'],say:'The Scribe multiplies the dream by the input dial: only the allowed amount gets written.'},
 {t:'The Basket makes the new memory',from:'kept memory + written content',to:'c(t) on the belt',k:['addc'],say:'The Basket **adds** the kept memory and the new writing. This is the new long-term memory.'},
 {t:'Output gate: the Mixer',from:'x and h(t−1)',to:'Mixer 3',k:['cat3'],say:'The fourth Mixer prepares the output gate.'},
 {t:'Output gate: the Dimmer switch',from:'Mixer 3',to:'Dimmer σ',k:['so'],say:'Dial **o** (0 to 1) says how much memory may be said out loud.'},
 {t:'The Squeezer calms the memory',from:'c(t) from the belt',to:'tanh(c)',k:['tanhc'],say:'The Squeezer tanh squashes the memory to −1…1 so it is safe to reveal.'},
 {t:'The Speaker says it',from:'dial o and tanh(c)',to:'h(t)',k:['mulh'],say:'The Speaker multiplies: only the allowed part of the memory becomes the output **h**.'},
 {t:'The outputs leave the cell',from:'the cell',to:'the next step and the output layer',k:['ct','hup'],say:'**c(t)** and **h(t)** travel on to the next word. **h(t)** also goes up to the output layer. One step of the LSTM is complete.'}];
function gStop(){if(A.g.play){clearInterval(A.g.play);A.g.play=null;}}
function gGo(s){A.g.s=Math.max(0,Math.min(GSTEPS.length-1,s));renderArch();}
function guidedHTML(){const g=GSTEPS[A.g.s],n=A.g.s;
  const dots=GSTEPS.map((_,i)=>`<i class="${i<n?'d':i===n?'a':''}" data-i="${i}"></i>`).join('');
  const body=g.k.map(k=>archInfo(k,A.t,true)).join('<hr class="sep">');
  return`<div class="gstep"><div class="gk">Step ${n+1} of ${GSTEPS.length} · time step t = ${A.t+1}</div><h3>${rt(g.t)}</h3>
   <div class="flow"><span class="fb">${esc(g.from)}</span><span class="fa">➜</span><span class="fb to">${esc(g.to)}</span></div>
   <p class="gsay">${rt(g.say)}</p>${INSIDE.has(g.k[0])?`<button class="btn lookb" data-look="${g.k[0]}">🔍 Look inside this part</button>`:''}${body}</div>`;}
function renderArch(){
  const root=$('#archroot');
  if(S.cfg.cell!=='lstm'){root.dataset.ready='';root.innerHTML=`<div class="note" style="padding:18px">The Architecture tab draws the <b>LSTM</b> cell. You are on GRU. <button class="btn pri" id="archgo">Switch to LSTM</button></div>`;$('#archgo').onclick=()=>$('#cellsel button[data-c=lstm]').click();return;}
  if(!root.dataset.ready){root.dataset.ready='1';
    root.innerHTML=`<div class="archbar"><div class="seg" id="a-run"><button data-r="0">Explore all</button><button data-r="1">▶ Step by step</button></div>
      <div class="seg" id="a-mode"><button data-m="cell">Inside one cell</button><button data-m="time">Unrolled in time</button></div>
      <div class="seg" id="a-t"></div>
      <div class="seg" id="a-foc"><button data-f="all">Everything</button><button data-f="forget">Forget</button><button data-f="input">Input</button><button data-f="cell">Memory update</button><button data-f="output">Output</button></div>
      <button class="btn" id="a-all">Expand all</button><label class="note" id="a-vl"><input type="checkbox" id="a-vals" checked> numbers</label><label class="note" id="a-pl"><input type="checkbox" id="a-peep"> peepholes</label>
      <span class="zoomb"><button class="btn" id="a-zm">−</button><button class="btn" id="a-fit">Fit</button><button class="btn" id="a-zp">+</button></span></div>
      <div class="gbarx" id="a-gbar"><button class="btn" id="g-prev">◀ Back</button><button class="btn pri" id="g-next">Next ▶</button><button class="btn" id="g-play">▶ Play</button><div class="gdots" id="g-dots"></div><button class="btn" id="g-reset">↺ Restart</button></div>
      <div class="archcap" id="a-cap"></div>
      <div class="archgrid"><div class="archstage" id="a-stage"><div class="archwrap" id="a-wrap"><svg id="a-svg"></svg></div><div class="insideov" id="a-ov"></div></div><aside class="archinfo" id="a-info"></aside></div>
      <div class="archleg"><span><i style="background:${AC.x}"></i>Input x</span><span><i style="background:${AC.belt}"></i>Memory c (the belt)</span><span><i style="background:${AC.h}"></i>Previous output h</span><span><i style="background:${AC.out}"></i>Output h</span><span><i style="background:${AC.sig}"></i>σ dimmer (0 to 1)</span><span><i style="background:${AC.tan}"></i>tanh (−1 to 1)</span><span><i style="background:${AC.op}"></i>× multiply</span><span><i style="background:${AC.box}"></i>+ mix / add</span><span><i style="background:${AC.bias}"></i>bias</span></div>`;
    $('#a-run').onclick=e=>{const b=e.target.closest('button');if(!b)return;gStop();A.g.on=b.dataset.r==='1';if(A.g.on){A.mode='cell';A.g.s=0;A.sel=null;}A.zoom=null;renderArch();};
    $('#a-mode').onclick=e=>{const b=e.target.closest('button');if(!b)return;A.mode=b.dataset.m;if(A.mode==='time'){gStop();A.g.on=false;}A.sel=null;A.zoom=null;renderArch();};
    $('#a-t').onclick=e=>{const b=e.target.closest('button');if(!b)return;A.t=+b.dataset.t;renderArch();};
    $('#a-foc').onclick=e=>{const b=e.target.closest('button');if(!b)return;A.focus=b.dataset.f;renderArch();};
    $('#a-all').onclick=()=>{const T=S.cfg.T;if(A.open.size>=T)A.open.clear();else for(let t=0;t<T;t++)A.open.add(t);A.zoom=null;renderArch();};
    $('#a-vals').onchange=e=>{A.vals=e.target.checked;renderArch();};$('#a-peep').onchange=e=>{A.peep=e.target.checked;renderArch();};
    $('#a-zp').onclick=()=>{A.zoom=(A.zoom||archFit())*1.2;renderArch();};$('#a-zm').onclick=()=>{A.zoom=(A.zoom||archFit())/1.2;renderArch();};$('#a-fit').onclick=()=>{A.zoom=null;renderArch();};
    $('#g-prev').onclick=()=>{gStop();gGo(A.g.s-1);};$('#g-next').onclick=()=>{gStop();gGo(A.g.s+1);};$('#g-reset').onclick=()=>{gStop();gGo(0);};
    $('#g-play').onclick=()=>{if(A.g.play){gStop();renderArch();return;}if(A.g.s>=GSTEPS.length-1)A.g.s=0;A.g.play=setInterval(()=>{if(A.g.s>=GSTEPS.length-1){gStop();renderArch();return;}gGo(A.g.s+1);},6500);gGo(A.g.s);};
    $('#g-dots').onclick=e=>{const d=e.target.closest('i');if(d){gStop();gGo(+d.dataset.i);}};
    document.addEventListener('keydown',e=>{if(tab!=='arch'||!A.g.on||/INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName))return;if(e.key==='ArrowRight'){gStop();gGo(A.g.s+1);}if(e.key==='ArrowLeft'){gStop();gGo(A.g.s-1);}});
    $('#a-svg').addEventListener('click',e=>{const sb=e.target.closest('[data-act="shrink"]');if(sb){A.open.delete(+sb.dataset.t);A.zoom=null;renderArch();return;}
      const g=e.target.closest('.part');if(!g)return;const t=+g.dataset.t;if(g.dataset.act==='expand'){A.open.add(t);A.t=t;A.zoom=null;renderArch();return;}const nm=g.dataset.p;if(!GS()){A.t=t;A.sel=nm;renderArch();}if(INSIDE.has(nm)){const el=document.querySelector(`#a-svg [data-p="${nm}"]`);openInside(nm,A.t,el);}});
    $('#a-ov').addEventListener('click',e=>{if(e.target.id==='a-ov')closeInside();});
    $('#a-info').addEventListener('click',e=>{const b=e.target.closest('[data-look]');if(b){const nm=b.dataset.look;openInside(nm,A.t,document.querySelector(`#a-svg [data-p="${nm}"]`));}});
    document.addEventListener('keydown',e=>{if(e.key==='Escape'&&A.inside)closeInside();});
    window.addEventListener('resize',()=>{if(tab==='arch'&&A.zoom==null)renderArch();});}
  A.m=forward(S.cfg);const T=S.cfg.T;if(A.t>=T)A.t=T-1;const gd=GS();const sg=[A.g.s,A.g.on,A.mode,A.t,A.focus,A.vals].join();if(A._sg!==sg){closeInside();A._sg=sg;}
  document.querySelectorAll('#a-run button').forEach(b=>b.classList.toggle('on',(b.dataset.r==='1')===A.g.on));
  document.querySelectorAll('#a-mode button').forEach(b=>b.classList.toggle('on',b.dataset.m===A.mode));
  document.querySelectorAll('#a-foc button').forEach(b=>b.classList.toggle('on',b.dataset.f===A.focus));
  $('#a-t').style.display=A.mode==='cell'?'':'none';$('#a-all').style.display=A.mode==='time'?'':'none';$('#a-all').textContent=A.open.size>=T?'Shrink all':'Expand all';
  $('#a-foc').style.display=gd?'none':'';$('#a-mode').style.display=gd?'none':'';$('#a-vl').style.display=$('#a-pl').style.display=gd?'none':'';
  $('#a-gbar').style.display=gd?'':'none';
  $('#a-t').innerHTML=[...Array(T).keys()].map(t=>`<button data-t="${t}" class="${t===A.t?'on':''}">t = ${t+1}</button>`).join('');
  $('#a-cap').style.display=gd?'none':'';$('#a-cap').innerHTML=rt(FCAP[A.focus]);
  if(gd){$('#g-dots').innerHTML=GSTEPS.map((_,i)=>`<i class="${i<A.g.s?'d':i===A.g.s?'a':''}" data-i="${i}" title="${esc(GSTEPS[i].t)}"></i>`).join('');$('#g-play').textContent=A.g.play?'⏸ Pause':'▶ Play';$('#g-prev').disabled=A.g.s===0;$('#g-next').disabled=A.g.s===GSTEPS.length-1;}
  const g=archSVG(),svg=$('#a-svg'),z=A.zoom||archFit(g.W,g.H);svg.setAttribute('viewBox',`0 0 ${g.W} ${g.H}`);svg.style.width=Math.round(g.W*z)+'px';svg.style.height=Math.round(g.H*z)+'px';svg.innerHTML=g.html;A._W=g.W;A._H=g.H;
  $('#a-info').innerHTML=gd?guidedHTML():(A.sel?archInfo(A.sel,A.t)+(INSIDE.has(A.sel)?`<button class="btn lookb" data-look="${A.sel}">🔍 Look inside this part</button>`:''):`<h3>Click any part</h3><p class="note">Every shape in the diagram is clickable: the inputs, the four mixers, the gates, the × and + operations, the memory belt and the outputs. You will see what it is, why it exists and its numbers for the chosen time step.</p><p class="note">Press <b>▶ Step by step</b> to watch the cell being built one operation at a time. Use <b>Unrolled in time</b> to see the cell repeated over the sequence.</p>`);
  document.querySelector('#archroot').classList.toggle('guided',gd);
}
/* ---------- "Look inside" view: what really happens inside a node ---------- */
const INSIDE=new Set(['cat0','cat1','cat2','cat3','sf','si','tg','so','mulf','muli','addc','tanhc','mulh']);
const UC=['#D9730D','#7C3AED','#16A34A','#0E7490','#B7791F','#C0392B'];
const uvec=v=>v.map((x,j)=>`<span class="uv" style="--uc:${UC[j%6]}"><i>unit ${j+1}</i>${N(x)}</span>`).join('');
function inCurve(kind,zs,ys){const W=360,Hh=200,p=34,xr=kind==='sig'?[-6,6]:[-3,3],yr=kind==='sig'?[0,1]:[-1,1];
  const X=z=>p+(Math.max(xr[0],Math.min(xr[1],z))-xr[0])/(xr[1]-xr[0])*(W-2*p),Y=y=>Hh-p-(y-yr[0])/(yr[1]-yr[0])*(Hh-2*p);
  const f=kind==='sig'?z=>1/(1+Math.exp(-z)):Math.tanh;let d='';for(let i=0;i<=120;i++){const z=xr[0]+(xr[1]-xr[0])*i/120;d+=(i?'L':'M')+X(z).toFixed(1)+' '+Y(f(z)).toFixed(1);}
  const y0=Y(kind==='sig'?0:0),ticks=(kind==='sig'?[-6,-3,0,3,6]:[-3,-1.5,0,1.5,3]).map(z=>`<text x="${X(z)}" y="${Hh-p+16}" text-anchor="middle" class="tk">${z}</text>`).join('');
  const yt=(kind==='sig'?[0,.5,1]:[-1,0,1]).map(y=>`<line x1="${p}" x2="${W-p}" y1="${Y(y)}" y2="${Y(y)}" class="gl2"/><text x="${p-6}" y="${Y(y)+4}" text-anchor="end" class="tk">${y}</text>`).join('');
  const dots=zs.map((z,j)=>{const c=UC[j%6],x=X(z),yy=Y(ys[j]),b=Y(0);return`<line x1="${x}" x2="${x}" y1="${b}" y2="${b}" stroke="${c}" stroke-width="2" stroke-dasharray="4 3"><animate attributeName="y2" from="${b}" to="${yy}" dur="1.6s" begin=".9s" fill="freeze"/></line>
   <line x1="${x}" x2="${x}" y1="${yy}" y2="${yy}" stroke="${c}" stroke-width="2" stroke-dasharray="4 3"><animate attributeName="x2" from="${x}" to="${p}" dur="1.2s" begin="2.5s" fill="freeze"/></line>
   <circle cx="${x}" cy="${b}" r="6.5" fill="${c}" stroke="#fff" stroke-width="2"><animate attributeName="cy" from="${b}" to="${yy}" dur="1.6s" begin=".9s" fill="freeze"/></circle>
   <circle cx="${p}" cy="${yy}" r="0" fill="${c}" stroke="#fff" stroke-width="2"><animate attributeName="r" from="0" to="6.5" dur=".4s" begin="3.7s" fill="freeze"/></circle>`}).join('');
  return`<svg class="icurve" viewBox="0 0 ${W} ${Hh}">${yt}<line x1="${p}" x2="${W-p}" y1="${Hh-p}" y2="${Hh-p}" class="ax"/><line x1="${p}" x2="${p}" y1="${p-8}" y2="${Hh-p}" class="ax"/>${ticks}<path d="${d}" fill="none" stroke="#475569" stroke-width="2.5"/>${dots}<text x="${W-p}" y="${p-12}" text-anchor="end" class="tk">number in (bottom) → number out (left)</text></svg>`;}
function insideHTML(nm,t){
  const cfg=S.cfg,M=A.m,P=cfg.p,x=cfg.x[t],hp=M.h[t],cp=M.c[t],G=M.G[t],ct=M.c[t+1],H=cfg.H;
  const gi=nm.startsWith('cat')?GI[+nm[3]]:nm==='sf'?'f':nm==='si'?'i':nm==='tg'?'g':nm==='so'?'o':null;
  let title,sym,symc,lede,inL,inV,outL,outV,fn,curve='',head,rows;
  const T=(h,r)=>`<table class="itab"><tr>${h.map(c=>`<th>${c}</th>`).join('')}</tr>${r.map((row,i)=>`<tr style="--i:${i}">${row.map(c=>`<td>${c}</td>`).join('')}</tr>`).join('')}</table>`;
  const dot=j=>`<b class="ud" style="background:${UC[j%6]}">${j+1}</b>`;
  if(nm.startsWith('cat')){const k=gi,w=WS[k],Pk=P[k],z=preAct(P,k,x,hp);
    title=`Inside the Mixer of the ${GN[k]}`;sym='+';symc=AC.box;lede='Each memory unit has its own row of weights. It multiplies every input number by its weight, adds everything up, then adds its bias.';
    inL='x(t) and h(t−1)';inV=null;outL=`z for the ${GN[k]}`;outV=z;
    fn=K(R`z_j=\sum_i W_{ji}x_i+\sum_i U_{ji}h_i+b_j`,true);
    head=['unit','weights × x','weights × h(t−1)','+ bias','= z'];
    rows=[...Array(H).keys()].map(j=>{const a=Pk.W[j].map((v,i)=>`${f2(v)}×${f2(x[i])}`).join(' + '),sa=Pk.W[j].reduce((s,v,i)=>s+v*x[i],0),b2=Pk.U[j].map((v,i)=>`${f2(v)}×${f2(hp[i])}`).join(' + '),sb=Pk.U[j].reduce((s,v,i)=>s+v*hp[i],0);
      return[dot(j),`${a}<br><b>= ${N(sa)}</b>`,`${b2}<br><b>= ${N(sb)}</b>`,N(Pk.b[j]),`<b>${N(z[j])}</b>`];});
    const xin=`<span class="vl">x(t)</span>${uvec(x)}<span class="vl">h(t−1)</span>${uvec(hp)}`;
    return card(title,sym,symc,lede,xin,fn,`<span class="vl">z</span>${uvec(z)}`,'',T(head,rows));}
  if(gi&&(nm==='sf'||nm==='si'||nm==='so')){const k=gi,z=preAct(P,k,x,hp),o=G[k];
    title=`Inside the Dimmer switch (σ) of the ${GN[k]}`;sym='σ';symc=AC.sig;lede='The dimmer takes any number and turns it into a dial between 0 (off) and 1 (fully on). The curve shows how: small numbers go near 0, big numbers go near 1.';
    fn=K(R`\sigma(z)=\dfrac{1}{1+e^{-z}}`,true);
    head=['unit','z (came in)','e<sup>−z</sup>','1 + e<sup>−z</sup>','1 ÷ that = dial'];
    rows=z.map((v,j)=>[dot(j),N(v),N(Math.exp(-v)),N(1+Math.exp(-v)),`<b>${N(o[j])}</b>`]);
    return card(title,sym,symc,lede,`<span class="vl">z from the Mixer</span>${uvec(z)}`,fn,`<span class="vl">${k} (dial)</span>${uvec(o)}`,inCurve('sig',z,o),T(head,rows));}
  if(nm==='tg'||nm==='tanhc'){const isg=nm==='tg',z=isg?preAct(P,'g',x,hp):ct,o=z.map(Math.tanh);
    title=isg?'Inside the Dreamer (tanh): the new idea':'Inside the Squeezer (tanh): calming the memory';sym='tanh';symc=AC.tan;lede=isg?'tanh squeezes any number into the range −1 to 1. Here it turns the mixed numbers into the content to remember.':'The memory can grow big. tanh squeezes it into the range −1 to 1 before it is revealed.';
    fn=K(R`\tanh(z)=\dfrac{e^{z}-e^{-z}}{e^{z}+e^{-z}}`,true);
    head=['unit',isg?'z (came in)':'c(t) (came in)','e<sup>z</sup>','e<sup>−z</sup>','(e<sup>z</sup>−e<sup>−z</sup>) ÷ (e<sup>z</sup>+e<sup>−z</sup>)'];
    rows=z.map((v,j)=>[dot(j),N(v),N(Math.exp(v)),N(Math.exp(-v)),`<b>${N(o[j])}</b>`]);
    return card(title,sym,symc,lede,`<span class="vl">${isg?'z from the Mixer':'c(t) from the belt'}</span>${uvec(z)}`,fn,`<span class="vl">${isg?'c̃':'tanh(c)'}</span>${uvec(o)}`,inCurve('tanh',z,o),T(head,rows));}
  if(nm==='mulf'||nm==='muli'||nm==='mulh'){const a=nm==='mulf'?G.f:nm==='muli'?G.i:G.o,b=nm==='mulf'?cp:nm==='muli'?G.g:ct.map(Math.tanh),an=nm==='mulf'?'f':nm==='muli'?'i':'o',bn=nm==='mulf'?'c(t−1)':nm==='muli'?'c̃':'tanh(c)',r=a.map((v,j)=>v*b[j]);
    title=nm==='mulf'?'Inside the Forget Guard (×)':nm==='muli'?'Inside the Scribe (×)':'Inside the Speaker (×)';sym='×';symc=AC.op;
    lede=nm==='mulf'?'For every memory unit, the guard multiplies the old memory by its dial. Dial 0 wipes it, dial 1 keeps it.':nm==='muli'?'The Scribe multiplies the new idea by the input dial: only the allowed amount is written.':'The Speaker multiplies the squeezed memory by the output dial: only the allowed part is said out loud.';
    fn=K((nm==='mulf'?R`\text{result}_j=f_j\times c_{t-1,j}`:nm==='muli'?R`\text{result}_j=i_j\times\tilde c_j`:R`\text{result}_j=o_j\times\tanh(c_j)`),true);
    const bar=(v,c)=>`<span class="mbar"><i style="width:${Math.min(1,Math.abs(v))*100}%;background:${c};${v<0?'opacity:.55':''}"></i></span>`;
    head=['unit',`${an} (dial)`,bn,`${an} × ${bn} = result`];
    rows=a.map((v,j)=>[dot(j),`${N(v)}${bar(v,'#F5B50A')}`,`${N(b[j])}${bar(b[j],AC.belt)}`,`<b>${N(r[j])}</b>${bar(r[j],AC.op)}`]);
    return card(title,sym,symc,lede,`<span class="vl">${an} (dial)</span>${uvec(a)}<span class="vl">${bn}</span>${uvec(b)}`,fn,`<span class="vl">result</span>${uvec(r)}`,'',T(head,rows));}
  if(nm==='addc'){const kp=G.f.map((v,j)=>v*cp[j]),wr=G.i.map((v,j)=>v*G.g[j]);
    title='Inside the Basket (+): the new memory';sym='+';symc=AC.box;lede='For every memory unit, the Basket adds what the guard kept to what the Scribe wrote. The sum is the new long-term memory.';
    fn=K(R`c_t=\underbrace{f_t\odot c_{t-1}}_{\text{kept}}+\underbrace{i_t\odot\tilde c_t}_{\text{written}}`,true);
    head=['unit','kept','+ written','= new memory c(t)'];
    rows=kp.map((v,j)=>[dot(j),N(v),N(wr[j]),`<b>${N(ct[j])}</b>`]);
    return card(title,sym,symc,lede,`<span class="vl">kept</span>${uvec(kp)}<span class="vl">written</span>${uvec(wr)}`,fn,`<span class="vl">c(t)</span>${uvec(ct)}`,'',T(head,rows));}
  return null;
}
function card(title,sym,symc,lede,inHTML,fn,outHTML,curve,table){
  return`<div class="ihead"><span class="isym" style="background:${symc};color:${symc===AC.box?'#3C5068':'#fff'}">${sym}</span><div><h3>${title}</h3><div class="ilede">${esc(lede)}</div></div></div>
  <div class="iflow"><div class="ibox"><div class="ik">What came in</div>${inHTML}</div><div class="iarr">➜</div><div class="ibox mid"><div class="ik">What happens inside</div>${fn}</div><div class="iarr">➜</div><div class="ibox"><div class="ik">What came out</div>${outHTML}</div></div>
  ${curve?`<div class="icwrap">${curve}</div>`:''}<div class="itw">${table}</div>`;
}
function openInside(nm,t,src){
  const html=insideHTML(nm,t);if(!html)return;const ov=$('#a-ov'),svg=$('#a-svg');
  ov.innerHTML=`<div class="insidecard"><button class="ix" id="a-ox">✕ Close</button>${html}</div>`;ov.classList.add('on');A.inside=nm;
  svg.classList.add('zooming');svg.querySelectorAll('.part.focus').forEach(e=>e.classList.remove('focus'));if(src)src.classList.add('focus');
  const card=ov.querySelector('.insidecard');
  if(src){const sr=src.getBoundingClientRect(),cr=card.getBoundingClientRect(),dx=sr.left+sr.width/2-(cr.left+cr.width/2),dy=sr.top+sr.height/2-(cr.top+cr.height/2),sc=Math.max(.06,Math.min(sr.width/cr.width,sr.height/cr.height));
    card.style.transition='none';card.style.transform=`translate(${dx}px,${dy}px) scale(${sc})`;card.style.opacity='0';card.offsetWidth;
    card.style.transition='transform 1.1s cubic-bezier(.2,.8,.2,1),opacity .7s ease';card.style.transform='none';card.style.opacity='1';}
  $('#a-ox').onclick=closeInside;
}
function closeInside(){const ov=$('#a-ov');if(!ov)return;ov.classList.remove('on');ov.innerHTML='';A.inside=null;const svg=$('#a-svg');if(svg){svg.classList.remove('zooming');svg.querySelectorAll('.part.focus').forEach(e=>e.classList.remove('focus'));}}

function archFit(W,H){W=W||A._W||960;H=H||A._H||470;const w=($('#a-wrap')&&$('#a-wrap').clientWidth)||900,hh=Math.max(360,innerHeight-250);return Math.max(.3,Math.min(1.25,(w-8)/W,hh/H));}
