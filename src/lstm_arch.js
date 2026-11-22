/* ================= Architecture tab: deck-style clickable LSTM cell ================= */
const AC={cin:'#8FD0E8',h:'#F5A862',x:'#3E3A3B',out:'#F26B5B',cell:'#8DA35A',sig:'#EBD96B',tan:'#80858B',op:'#7DBDEB',box:'#DDEBF3',bias:'#F57C89',edge:'#3E3A3B'};
const A={mode:'cell',t:0,open:new Set(),sel:null,focus:'all',zoom:null,vals:true,peep:false,m:null};
const FOC={forget:'f',input:'i',cell:'u',output:'o'};
const FCAP={all:'Every part of the cell. Click any shape to see what it is and its live numbers.',
 forget:'**Forget gate.** Controls what information to throw away from memory: $f_t=\\sigma(W_f[h_{t-1},x_t]+b_f)$.',
 input:'**Input gate + candidate.** How much of the new information is written: $i_t=\\sigma(W_i[h_{t-1},x_t]+b_i)$ and $\\tilde c_t=\\tanh(W_c[h_{t-1},x_t]+b_c)$.',
 cell:'**Memory (cell state) update.** The cell state forgets some information, then is updated by adding new information: $c_t=f_t\\odot c_{t-1}+i_t\\odot\\tilde c_t$.',
 output:'**Output gate + hidden state.** A filtered version of the cell state becomes the new output: $o_t=\\sigma(W_o[h_{t-1},x_t]+b_o)$, $h_t=o_t\\odot\\tanh(c_t)$.'};
const MK={h:AC.h,x:AC.x,c:AC.cin,g:AC.cell,e:AC.edge,o:AC.out};
const CXG={f:190,i:330,g:470,o:560},GI=['f','i','g','o'];
const vs=v=>'['+v.slice(0,3).map(f2).join(', ')+(v.length>3?', …':'')+']';
const dm=tags=>(A.focus==='all'||tags.split(' ').includes(FOC[A.focus]))?'':'dim';
const Ln=(pts,mk,tags,arrow=true,w=3)=>`<path class="${dm(tags)}" d="M${pts.map(p=>p.join(' ')).join(' L')}" fill="none" stroke="${MK[mk]}" stroke-width="${w}" stroke-linejoin="round"${arrow?` marker-end="url(#ah-${mk})"`:''}/>`;
const lab=(b,s)=>`${b}<tspan dy="3" font-size="9">${s}</tspan>`;
const circ=(x,y,r,fill,tags,txt,col='#fff',fs=13)=>`<g class="${dm(tags)}"><circle cx="${x}" cy="${y}" r="${r}" fill="${fill}"/><text x="${x}" y="${y+4}" text-anchor="middle" font-size="${fs}" font-weight="600" fill="${col}">${txt}</text></g>`;
const chip=(t,x,y,anc='middle')=>A.vals?`<text class="chip" x="${x}" y="${y}" text-anchor="${anc}">${t}</text>`:'';
const PT=(nm,t,title,inner)=>`<g class="part${A.sel===nm&&A.t===t?' sel':''}" data-p="${nm}" data-t="${t}"><title>${title}</title>${inner}</g>`;
const sigNode=(x,y,tags)=>`<g class="${dm(tags)}"><circle cx="${x}" cy="${y}" r="22" fill="${AC.sig}"/><path d="M${x+22} ${y} A22 22 0 0 0 ${x+6} ${y-21} L${x+22} ${y-21}Z" fill="#fff" opacity=".0"/><text x="${x}" y="${y+8}" text-anchor="middle" font-size="24" font-family="Georgia,serif" font-style="italic" fill="#3b3b3b">σ</text></g>`;
const tanNode=(x,y,tags)=>{const p=[[0,-25],[24,-8],[15,20],[-15,20],[-24,-8]].map(q=>`${x+q[0]},${y+q[1]}`).join(' ');return`<g class="${dm(tags)}"><polygon points="${p}" fill="${AC.tan}"/><text x="${x}" y="${y+5}" text-anchor="middle" font-size="11" fill="#fff" font-weight="600">tanh</text></g>`;};
const mulNode=(x,y,tags)=>`<g class="${dm(tags)}"><circle cx="${x}" cy="${y}" r="13" fill="${AC.op}"/><text x="${x}" y="${y+6}" text-anchor="middle" font-size="19" fill="#1f3b52">×</text></g>`;
const addBox=(x,y,w,tags)=>`<g class="${dm(tags)}"><rect x="${x-w/2}" y="${y-13}" width="${w}" height="26" rx="4" fill="${AC.box}" stroke="#9DB8C8"/><text x="${x}" y="${y+7}" text-anchor="middle" font-size="20" fill="#4a5b66">+</text></g>`;

function expCell(t,ox,left,right,showShrink){
  const cfg=S.cfg,M=A.m,xl=left?45:0,xr=right?850:880,G=M.G[t],xv=cfg.x[t],hp=M.h[t],cp=M.c[t],ct=M.c[t+1],ht=M.h[t+1],tn=ct.map(Math.tanh);
  const tt=t+1;let s=`<g transform="translate(${ox},0)">`;
  s+=`<rect class="cellbox" x="100" y="40" width="700" height="360" rx="38"/>`;
  s+=`<text class="cap" x="120" y="388">LSTM cell · t = ${tt}</text>`;
  if(showShrink)s+=`<g class="shrinkbtn" data-act="shrink" data-t="${t}"><rect x="690" y="368" width="94" height="22" rx="11"/><text x="737" y="383" text-anchor="middle">– shrink</text></g>`;
  /* peepholes (optional) */
  if(A.peep){s+=`<g class="peep">`+Ln([[120,70],[120,262],[583,262]],'c','peep',false,2)+GI.map(k=>Ln([[CXG[k]-20,262],[CXG[k]-20,277]],'c','peep',true,2)).join('')+`<text x="126" y="256" font-size="10" fill="#4a7b8c">peephole links (not used in this lab's maths)</text></g>`;}
  /* hidden-state bus + arrows into the four combiners */
  s+=PT('hprev',t,'h(t-1): previous output',Ln([[left?61:0,345],[583,345]],'h','f i u o',false)+GI.map(k=>Ln([[CXG[k]-24,345],[CXG[k]-24,303]],'h','f i u o')).join('')+(left?circ(45,345,16,AC.h,'f i u o',lab('h','t-1')):'')+(left?chip(vs(hp),45,375):''));
  /* input bus */
  s+=PT('x',t,'x(t): current input',Ln([[330,422],[330,372]],'x','f i u o',false)+Ln([[190,372],[560,372]],'x','f i u o',false)+GI.map(k=>Ln([[CXG[k],372],[CXG[k],303]],'x','f i u o')).join('')+circ(330,440,18,AC.x,'f i u o',lab('x','t'))+chip(vs(xv),358,444,'start'));
  /* four combiners + biases */
  GI.forEach((k,n)=>{const tg=k==='f'?'f u':k==='o'?'o':'i u',cx=CXG[k];
    s+=PT('cat'+n,t,`W·[h(t-1), x(t)] for the ${GN[k]}`,addBox(cx,290,66,tg));
    s+=PT('b'+n,t,`bias b of the ${GN[k]}`,Ln([[cx+24,312],[cx+24,303]],'e',tg,true,2)+circ(cx+24,322,10,AC.bias,tg,n,'#fff',11));});
  /* cell-state highway */
  s+=PT('cprev',t,'c(t-1): memory from the previous step',Ln([[left?61:0,70],[177,70]],'c','f u',true)+(left?circ(45,70,16,AC.cin,'f u',lab('c','t-1'),'#1f3b52')+chip(vs(cp),45,42):''));
  s+=PT('cline',t,'cell state: the conveyor belt',Ln([[203,70],[441,70]],'g','u',true)+Ln([[499,70],[right?832:xr,70]],'g','u o',!right||true));
  s+=PT('mulf',t,'forget: f ⊙ c(t-1)',mulNode(190,70,'f u'));
  s+=PT('addc',t,'add new memory: + i ⊙ c̃',addBox(470,70,54,'u'));
  /* forget path */
  s+=PT('sf',t,'forget gate σ',Ln([[190,277],[190,239]],'e','f u')+sigNode(190,215,'f u')+Ln([[190,193],[190,85]],'e','f u')+chip('f = '+vs(G.f),198,178,'start'));
  /* input path */
  s+=PT('si',t,'input gate σ',Ln([[330,277],[330,239]],'e','i u')+sigNode(330,215,'i u')+Ln([[330,193],[330,140],[455,140]],'e','i u')+chip('i = '+vs(G.i),338,178,'start'));
  s+=PT('tg',t,'candidate tanh',Ln([[470,277],[470,243]],'e','i u')+tanNode(470,215,'i u')+Ln([[470,190],[470,155]],'e','i u')+chip('c̃ = '+vs(G.g),478,178,'start'));
  s+=PT('muli',t,'i ⊙ c̃',mulNode(470,140,'i u')+Ln([[470,127],[470,85]],'e','i u'));
  /* output path */
  s+=PT('so',t,'output gate σ',Ln([[560,277],[560,239]],'e','o')+sigNode(560,215,'o')+Ln([[582,215],[665,215]],'e','o')+chip('o = '+vs(G.o),568,178,'start'));
  s+=PT('tanhc',t,'tanh(c(t))',Ln([[680,70],[680,117]],'g','o',false)+tanNode(680,140,'o')+Ln([[680,165],[680,201]],'e','o')+chip('tanh(c)',715,140,'start'));
  s+=PT('mulh',t,'h = o ⊙ tanh(c)',mulNode(680,215,'o'));
  s+=PT('hup',t,'h(t) goes up to the output layer',Ln([[680,228],[680,345],[740,345],[740,43]],'o','o',true)+circ(740,25,16,AC.out,'o',lab('h','t'))+chip(vs(ht),770,28,'start'));
  s+=PT('hout',t,'h(t): output to the next step',(right?Ln([[740,345],[832,345]],'o','o',true)+circ(850,345,16,AC.out,'o',lab('h','t'))+chip(vs(ht),850,375):Ln([[740,345],[880,345]],'o','o',true)));
  s+=PT('ct',t,'c(t): new memory',right?circ(850,70,16,AC.cell,'u o',lab('c','t'))+chip(vs(ct),850,42):'');
  if(!right)s+=chip('c'+tt+' = '+vs(ct),860,56,'end');
  return s+'</g>';
}
function cmpCell(t,ox,left,right){
  const M=A.m,ct=M.c[t+1],ht=M.h[t+1],tt=t+1,cp=M.c[t],hp=M.h[t],xv=S.cfg.x[t];let s=`<g transform="translate(${ox},0)">`;
  s+=`<g class="part cmp${A.t===t?' sel':''}" data-act="expand" data-t="${t}"><title>Click to expand this cell</title><rect class="cellbox" x="50" y="40" width="140" height="360" rx="26"/>`+
   `<text x="120" y="190" text-anchor="middle" font-size="24" font-weight="700" fill="#33414d">LSTM</text><text x="120" y="218" text-anchor="middle" font-size="16" fill="#51606c">t = ${tt}</text>`+
   `<text x="120" y="262" text-anchor="middle" font-size="12" fill="#1E5873">▸ click to expand</text></g>`;
  s+=Ln([[left?36:0,70],[50,70]],'c','f u',false,3)+Ln([[190,70],[right?204:240,70]],'g','u o',true,3)+(left?circ(20,70,16,AC.cin,'f u',lab('c','t-1'),'#1f3b52')+chip(vs(cp),20,42):'');
  s+=Ln([[left?36:0,345],[50,345]],'h','f i u o',true,3)+Ln([[190,345],[right?204:240,345]],'o','o',true,3)+(left?circ(20,345,16,AC.h,'f i u o',lab('h','t-1'))+chip(vs(hp),20,375):'');
  s+=Ln([[120,422],[120,400]],'x','f i u o',true,3)+circ(120,440,18,AC.x,'f i u o',lab('x','t'))+chip(vs(xv),146,444,'start');
  s+=Ln([[150,40],[150,43]],'o','o',false,3)+Ln([[150,40],[150,43]],'o','o',false,3)+circ(150,25,16,AC.out,'o',lab('h','t'))+chip(vs(ht),176,28,'start');
  if(right)s+=circ(220,70,16,AC.cell,'u o',lab('c','t'))+chip(vs(ct),220,42)+circ(220,345,16,AC.out,'o',lab('h','t'))+chip(vs(ht),220,375);
  else s+=chip('c'+tt+' = '+vs(ct),230,56,'end');
  return s+'</g>';
}
function archSVG(){
  const cfg=S.cfg,T=cfg.T,times=A.mode==='cell'?[A.t]:[...Array(T).keys()];let off=30,body='';
  times.forEach((t,n)=>{const exp=A.mode==='cell'||A.open.has(t),left=n===0,right=n===times.length-1,w=exp?880:240;
    body+=exp?expCell(t,off,left,right,A.mode==='time'):cmpCell(t,off,left,right);off+=w;});
  const W=off+40,Hh=485;
  const defs=`<defs>${Object.entries(MK).map(([k,c])=>`<marker id="ah-${k}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0L10,5L0,10z" fill="${c}"/></marker>`).join('')}</defs>`;
  return{W,H:Hh,html:`${defs}${body}`};
}
function archInfo(nm,t){
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
  return`<h3>${rt(title)} <span class="tbadge">t = ${tt}</span></h3><p>${rt(what)}</p>${tex.map(D).join('')}${read?`<div class="read">${esc(read)}</div>`:''}${why?`<p class="why"><b>Why it exists:</b> ${rt(why)}</p>`:''}`;
}
function renderArch(){
  const root=$('#archroot');
  if(S.cfg.cell!=='lstm'){root.innerHTML=`<div class="note" style="padding:18px">The Architecture tab draws the <b>LSTM</b> cell exactly as in the lecture slides. You are on GRU. <button class="btn pri" id="archgo">Switch to LSTM</button></div>`;$('#archgo').onclick=()=>$('#cellsel button[data-c=lstm]').click();return;}
  if(!root.dataset.ready){root.dataset.ready='1';
    root.innerHTML=`<div class="archbar"><div class="seg" id="a-mode"><button data-m="cell">Inside one cell</button><button data-m="time">Unrolled in time</button></div>
      <div class="seg" id="a-t"></div>
      <div class="seg" id="a-foc"><button data-f="all">Everything</button><button data-f="forget">Forget</button><button data-f="input">Input</button><button data-f="cell">Memory update</button><button data-f="output">Output</button></div>
      <button class="btn" id="a-all">Expand all</button><label class="note"><input type="checkbox" id="a-vals" checked> numbers</label><label class="note"><input type="checkbox" id="a-peep"> peepholes</label>
      <span class="zoomb"><button class="btn" id="a-zm">−</button><button class="btn" id="a-fit">Fit</button><button class="btn" id="a-zp">+</button></span></div>
      <div class="archcap" id="a-cap"></div>
      <div class="archgrid"><div class="archwrap" id="a-wrap"><svg id="a-svg"></svg></div><aside class="archinfo" id="a-info"></aside></div>
      <div class="archleg"><span><i style="background:${AC.x}"></i>Input x</span><span><i style="background:${AC.cin}"></i>Memory c (cell state)</span><span><i style="background:${AC.h}"></i>Previous output h</span><span><i style="background:${AC.out}"></i>Output h</span><span><i style="background:${AC.sig}"></i>σ sigmoid (0 to 1)</span><span><i style="background:${AC.tan}"></i>tanh (−1 to 1)</span><span><i style="background:${AC.op}"></i>× multiply</span><span><i style="background:${AC.box}"></i>+ add / combine</span><span><i style="background:${AC.bias}"></i>bias</span></div>`;
    $('#a-mode').onclick=e=>{const b=e.target.closest('button');if(!b)return;A.mode=b.dataset.m;A.sel=null;A.zoom=null;renderArch();};
    $('#a-t').onclick=e=>{const b=e.target.closest('button');if(!b)return;A.t=+b.dataset.t;renderArch();};
    $('#a-foc').onclick=e=>{const b=e.target.closest('button');if(!b)return;A.focus=b.dataset.f;renderArch();};
    $('#a-all').onclick=()=>{const T=S.cfg.T;if(A.open.size>=T)A.open.clear();else for(let t=0;t<T;t++)A.open.add(t);A.zoom=null;renderArch();};
    $('#a-vals').onchange=e=>{A.vals=e.target.checked;renderArch();};$('#a-peep').onchange=e=>{A.peep=e.target.checked;renderArch();};
    $('#a-zp').onclick=()=>{A.zoom=(A.zoom||archFit())*1.2;renderArch();};$('#a-zm').onclick=()=>{A.zoom=(A.zoom||archFit())/1.2;renderArch();};$('#a-fit').onclick=()=>{A.zoom=null;renderArch();};
    $('#a-svg').addEventListener('click',e=>{const sb=e.target.closest('[data-act="shrink"]');if(sb){A.open.delete(+sb.dataset.t);A.zoom=null;renderArch();return;}
      const g=e.target.closest('.part');if(!g)return;const t=+g.dataset.t;if(g.dataset.act==='expand'){A.open.add(t);A.t=t;A.zoom=null;renderArch();return;}A.t=t;A.sel=g.dataset.p;renderArch();});
    window.addEventListener('resize',()=>{if(tab==='arch'&&A.zoom==null)renderArch();});}
  A.m=forward(S.cfg);const T=S.cfg.T;if(A.t>=T)A.t=T-1;
  document.querySelectorAll('#a-mode button').forEach(b=>b.classList.toggle('on',b.dataset.m===A.mode));
  document.querySelectorAll('#a-foc button').forEach(b=>b.classList.toggle('on',b.dataset.f===A.focus));
  $('#a-t').style.display=A.mode==='cell'?'':'none';$('#a-all').style.display=A.mode==='time'?'':'none';$('#a-all').textContent=A.open.size>=T?'Shrink all':'Expand all';
  $('#a-t').innerHTML=[...Array(T).keys()].map(t=>`<button data-t="${t}" class="${t===A.t?'on':''}">t = ${t+1}</button>`).join('');
  $('#a-cap').innerHTML=rt(FCAP[A.focus]);
  const g=archSVG(),svg=$('#a-svg'),z=A.zoom||archFit(g.W);svg.setAttribute('viewBox',`0 0 ${g.W} ${g.H}`);svg.style.width=Math.round(g.W*z)+'px';svg.style.height=Math.round(g.H*z)+'px';svg.innerHTML=g.html;A._W=g.W;
  $('#a-info').innerHTML=A.sel?archInfo(A.sel,A.t):`<h3>Click any part</h3><p class="note">Every shape in the diagram is clickable: the inputs, the four gates, the × and + operations, the memory line and the outputs. You will see what it is, why it exists and its numbers for the chosen time step.</p><p class="note">Use <b>Unrolled in time</b> to see the cell as a block repeated over the sequence. Click a block to expand it and <b>– shrink</b> to fold it back.</p>`;
}
function archFit(W){W=W||A._W||960;const w=($('#a-wrap')&&$('#a-wrap').clientWidth)||900;return Math.max(.3,Math.min(1.25,(w-6)/W));}
