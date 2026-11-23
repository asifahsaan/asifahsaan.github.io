/* ================= RNN Architecture tab: clickable cell, guided forward/backward, unrolled in time ================= */
const AC={h:'#2B7A9B',x:'#475569',out:'#7C3AED',sig:'#F5B50A',tan:'#10B981',op:'#3B82F6',box:'#E8EEF6',bias:'#F472B6',edge:'#334155',g:'#C0392B',loss:'#B7791F',gray:'#94A3B8'};
const RA={mode:'cell',t:0,open:new Set(),sel:null,zoom:null,vals:true,g:{on:false,back:false,s:0,play:null},m:null,gr:null,inside:null};
const MK={x:AC.x,h:AC.h,o:AC.out,e:AC.edge,g:AC.g,l:AC.loss};
const vs=v=>'['+v.slice(0,3).map(f2).join(', ')+(v.length>3?', …':'')+']';
const GS=()=>RA.g.on&&RA.mode==='cell';
const oHere=t=>outSteps(S.cfg).includes(t);
const RSTEPS=[
 {id:0,t:'The inputs arrive',from:'the new input and the memory of the last step',to:'the RNN cell',k:['x','hprev'],say:'**x** is the new input (a number vector, for example a letter or a word). **h(t−1)** is what the network remembers from the previous step. At the first step the memory is all zeros.'},
 {id:1,t:'Input weights read the new input',from:'x(t)',to:'W_xh · x',k:['mx'],say:'The matrix **W_xh** decides how much each input number matters. It multiplies the input and gives one number per hidden unit.'},
 {id:2,t:'Memory weights carry the old memory',from:'h(t−1)',to:'W_hh · h',k:['mh'],say:'The matrix **W_hh** decides how the old memory is passed on. This is the loop that makes the network "recurrent". The same W_hh is used at every step.'},
 {id:3,t:'Add everything and the bias',from:'W_xh·x, W_hh·h and the bias b',to:'s(t)',k:['sum'],say:'The two results and the bias are **added**. This gives the pre-activation **s**: the raw new memory before it is squashed.'},
 {id:4,t:'The activation squashes it',from:'s(t)',to:'activation',k:['act'],say:'The activation function bends the sum so the network can learn more than straight lines. For tanh every number is squeezed into −1…1.'},
 {id:5,t:'The new memory is ready',from:'the activation',to:'h(t)',k:['ht'],say:'**h(t)** is the new hidden state: the memory of everything read so far, including this step.'},
 {id:6,t:'The memory is sent on to the next step',from:'h(t)',to:'the next RNN cell',k:['hout'],say:'**h(t)** travels to the next step, where it becomes h(t−1). The next cell uses exactly the same weights.'},
 {id:7,t:'The output layer reads the memory',from:'h(t)',to:'W_hy·h + b_y',k:['mo'],say:'A separate weight matrix **W_hy** turns the memory into one raw score **u** for each possible answer.',out:1},
 {id:8,t:'The output function turns scores into answers',from:'u(t)',to:'output function',k:['oact'],say:'The output function converts the raw scores into the final answer **y** (probabilities for softmax, values between 0 and 1 for sigmoid).',out:1},
 {id:9,t:'The prediction y(t)',from:'the output function',to:'y(t)',k:['y'],say:'**y(t)** is what the network predicts at this step.',out:1},
 {id:10,t:'Compare with the target: the loss',from:'y(t) and the target',to:'loss L(t)',k:['loss'],say:'The loss measures how wrong the prediction is. This one number is what training tries to make smaller.',out:1}];
const BSTEPS=[
 {id:0,t:'The loss tells the output how wrong it is',from:'loss',to:'δy = y − target',k:'b0',say:'Going backward starts at the loss. Its first message is simply the prediction minus the target.',out:1},
 {id:1,t:'Through the output function',from:'δy',to:'δu',k:'b1',say:'The error is passed back through the output function (multiplied by its slope where needed).',out:1},
 {id:2,t:'Through the output layer',from:'δu',to:'∇W_hy, ∇b_y and δh',k:'b2',say:'The output layer weights get their gradient, and the error continues toward the hidden state.',out:1},
 {id:3,t:'Collect the error at the hidden state',from:'the output layer and the next step',to:'δh(t)',k:'b3',say:'The hidden state is used twice: by the output and by the next step. So its error is the **sum** of both messages. This is "through time".'},
 {id:4,t:'Through the activation',from:'δh(t)',to:'δs(t)',k:'b4',say:'The error is multiplied by the activation\'s slope. For tanh the slope is at most 1, so the error can only shrink here.'},
 {id:5,t:'Gradients of the shared weights',from:'δs(t)',to:'∇W_xh, ∇W_hh, ∇b',k:'b5',say:'Every weight learns from this step. The same weights are used at every step, so their final gradient is the **sum over all steps**.'},
 {id:6,t:'The error travels one step back in time',from:'δs(t)',to:'δh(t−1)',k:'b6',say:'The error is multiplied by W_hh and handed to the previous step. This happens again and again, so over a long sequence it shrinks (vanishing) or grows (exploding).'}];
const FSEQ=()=>RSTEPS.filter(s=>!s.out||oHere(RA.t));const BSEQ=()=>BSTEPS.filter(s=>!s.out||oHere(RA.t));
const SEQ=()=>RA.g.back?BSEQ():FSEQ();
const curId=()=>GS()?SEQ()[Math.min(RA.g.s,SEQ().length-1)].id:null;
const V=rv=>{if(!GS()||rv==null)return'';if(RA.g.back)return'fade';const c=curId();return rv>c?'hid':rv===c?'enter':'';};
const VB=rb=>{if(!GS()||!RA.g.back)return'hid';const c=curId();return rb>c?'hid':rb===c?'enter':'';};
const Ln=(pts,mk,rv,arrow=true,w=3,xc='')=>{const d=`M${pts.map(p=>p.join(' ')).join(' L')}`,c=(V(rv)+' '+xc).trim();
  return `<path class="ln ${c}" pathLength="1" d="${d}" fill="none" stroke="${MK[mk]}" stroke-width="${w}" stroke-linecap="round" stroke-linejoin="round"${arrow?` marker-end="url(#ah-${mk})"`:''}/>`+(V(rv)==='enter'?`<path class="flowdash" pathLength="1" d="${d}" fill="none" stroke-width="${w}" stroke-linecap="round" stroke-linejoin="round"/>`:'');};
const Gr=(pts,rb,w=3.5)=>{if(!(GS()&&RA.g.back))return'';const d=`M${pts.map(p=>p.join(' ')).join(' L')}`,c=VB(rb);
  return `<path class="ln grad ${c}" pathLength="1" d="${d}" fill="none" stroke="${AC.g}" stroke-width="${w}" stroke-linecap="round" stroke-linejoin="round" marker-end="url(#ah-g)"/>`+(c==='enter'?`<path class="flowdash" pathLength="1" d="${d}" fill="none" stroke-width="${w}" stroke-linecap="round" stroke-linejoin="round"/>`:'');};
const lab=(b,s)=>`${b}<tspan dy="3" font-size="9">${s}</tspan>`;
const circ=(x,y,r,fill,rv,txt,col='#fff',fs=13)=>`<g class="nd ${V(rv)}"><circle cx="${x}" cy="${y}" r="${r}" fill="${fill}" stroke="#fff" stroke-width="3"/><text x="${x}" y="${y+4}" text-anchor="middle" font-size="${fs}" font-weight="700" fill="${col}">${txt}</text></g>`;
const chipH=(t,x,y,anc,cl,g)=>{const w=t.length*6.3+12,x0=anc==='start'?x:anc==='end'?x-w:x-w/2;return`<g class="${cl}"><rect class="${g?'chipbg gbg':'chipbg'}" x="${x0}" y="${y-13}" width="${w}" height="19" rx="9.5"/><text class="chip${g?' gtx':''}" x="${x0+w/2}" y="${y}" text-anchor="middle">${t}</text></g>`;};
const chip=(t,x,y,anc,rv)=>RA.vals&&!(GS()&&RA.g.back&&rv!=null)?chipH(t,x,y,anc,V(rv),false):'';
const gchip=(t,x,y,anc,rb)=>GS()&&RA.g.back?chipH(t,x,y,anc,VB(rb),true):'';
const PT=(nm,t,title,inner,ex='')=>`<g class="part ${ex}${RA.sel===nm&&RA.t===t&&!GS()?' sel':''}" data-p="${nm}" data-t="${t}"><title>${title}</title>${inner}</g>`;
const rbox=(x,y,w,h,txt,fill,rv)=>`<g class="nd ${V(rv)}"><rect x="${x-w/2}" y="${y-h/2}" width="${w}" height="${h}" rx="11" fill="${fill}" stroke="#fff" stroke-width="3"/><text x="${x}" y="${y+5}" text-anchor="middle" font-size="13" font-weight="700" fill="#fff">${txt}</text></g>`;
const sigN=(x,y,rv)=>`<g class="nd ${V(rv)}"><circle cx="${x}" cy="${y}" r="23" fill="${AC.sig}" stroke="#fff" stroke-width="3"/><text x="${x}" y="${y+8}" text-anchor="middle" font-size="25" font-family="Georgia,serif" font-style="italic" fill="#3b2f00">σ</text></g>`;
const actN=(x,y,rv)=>{const a=S.cfg.act;return a==='sigmoid'?sigN(x,y,rv):`<g class="nd ${V(rv)}"><rect x="${x-27}" y="${y-21}" width="54" height="42" rx="14" fill="${a==='tanh'?AC.tan:'#F97316'}" stroke="#fff" stroke-width="3"/><text x="${x}" y="${y+5}" text-anchor="middle" font-size="${a==='tanh'?13:12}" font-weight="700" fill="#fff">${a==='tanh'?'tanh':'ReLU'}</text></g>`;};
const oactN=(x,y,rv)=>{const o=S.cfg.out;return o==='sigmoid'?sigN(x,y,rv):`<g class="nd ${V(rv)}"><rect x="${x-36}" y="${y-19}" width="72" height="38" rx="12" fill="${o==='softmax'?AC.sig:AC.gray}" stroke="#fff" stroke-width="3"/><text x="${x}" y="${y+5}" text-anchor="middle" font-size="13" font-weight="700" fill="${o==='softmax'?'#3b2f00':'#fff'}">${o==='softmax'?'softmax':'identity'}</text></g>`;};
const rTv=(M,v)=>M[0].map((_,j)=>M.reduce((s,row,c)=>s+row[j]*v[c],0));
function tokName(cfg,i){return cfg.tok&&cfg.tok.vocab?cfg.tok.vocab[i]:null;}
function xLabel(t){const cfg=S.cfg,n=cfg.tok?tokName(cfg,cfg.tok.seq[t]):null;return(n?`“${n}” `:'')+vs(cfg.x[t]);}
function tLabel(t){const cfg=S.cfg,k=outSteps(cfg).indexOf(t),n=cfg.tok?tokName(cfg,cfg.tok.seq[t+1]):null;return(n&&cfg.mode==='all'?`“${n}” `:n&&k===outSteps(cfg).length-1?`“${n}” `:'')+vs(cfg.t[k]);}
function expRCell(t,ox,left,right,shrink){
  const cfg=S.cfg,m=RA.m,gr=RA.gr,x=cfg.x[t],hp=m.h[t],ht=m.h[t+1],sv=m.s[t],oh=oHere(t),noo=oh?'':'dim',tt=t+1,k=outSteps(cfg).indexOf(t);
  const xa=mv(cfg.Wxh,x),ha=mv(cfg.Whh,hp),gd=GS(),bk=gd&&RA.g.back;let s=`<g transform="translate(${ox},0)">`;
  s+=`<rect class="cellbox" x="90" y="40" width="615" height="350" rx="34"/>`;
  s+=`<text class="cap" x="108" y="378">RNN cell · t = ${tt}${oh?'':' · no output at this step'}</text>`;
  if(shrink)s+=`<g class="shrinkbtn" data-act="shrink" data-t="${t}"><rect x="590" y="356" width="94" height="22" rx="11"/><text x="637" y="371" text-anchor="middle">– shrink</text></g>`;
  /* inputs */
  s+=PT('x',t,'x(t): current input',circ(300,445,19,AC.x,0,lab('x','t'))+chip(xLabel(t),330,449,'start',0));
  s+=PT('hprev',t,'h(t-1): memory from the previous step',(left?circ(45,250,17,AC.h,0,lab('h','t-1'))+chip(vs(hp),45,284,'middle',0):'')+'');
  /* two weight products */
  s+=PT('mx',t,'W_xh · x',Ln([[300,426],[300,369]],'x',1)+rbox(300,350,92,36,'W<tspan font-size="9" dy="3">xh</tspan><tspan dy="-3"> · x</tspan>',AC.op,1)+Ln([[300,331],[300,269]],'x',3)+chip('W·x = '+vs(xa),248,356,'end',1));
  s+=PT('mh',t,'W_hh · h',Ln([[left?62:0,250],[124,250]],'h',2)+rbox(170,250,92,36,'W<tspan font-size="9" dy="3">hh</tspan><tspan dy="-3"> · h</tspan>',AC.op,2)+Ln([[216,250],[283,250]],'h',3)+chip('W·h = '+vs(ha),170,214,'middle',2));
  s+=PT('sum',t,'s = W_xh·x + W_hh·h + b',`<g class="nd ${V(3)}"><circle cx="300" cy="250" r="16" fill="${AC.box}" stroke="#A9BBD0" stroke-width="2"/><text x="300" y="258" text-anchor="middle" font-size="22" font-weight="700" fill="#3C5068">+</text></g>`+chip('s = '+vs(sv),322,224,'start',3));
  s+=PT('bias',t,'bias b',circ(388,250,11,AC.bias,3,'b','#fff',11)+Ln([[377,250],[318,250]],'e',3,true,2));
  /* activation and new memory */
  s+=PT('act',t,'activation',Ln([[300,234],[300,184]],'e',4)+actN(300,160,4));
  s+=PT('ht',t,'h(t): the new memory',Ln([[300,137],[300,89]],'e',5)+circ(300,70,17,AC.h,5,lab('h','t'))+chip('h = '+vs(ht),326,54,'start',5));
  s+=PT('hout',t,'h(t) goes to the next step',Ln([[318,70],[right?722:780,70]],'h',6)+(right?circ(740,70,17,AC.h,6,lab('h','t'))+chip(vs(ht),740,102,'middle',6):''));
  /* output layer */
  s+=PT('mo',t,'W_hy · h + b_y',Ln([[480,70],[480,128]],'o',7)+rbox(480,150,118,36,'W<tspan font-size="9" dy="3">hy</tspan><tspan dy="-3"> · h + b</tspan>',AC.op,7)+(oh?chip('u = '+vs(m.u[t]),544,154,'start',7):''),noo);
  s+=PT('oact',t,'output function',Ln([[480,168],[480,208]],'e',8)+oactN(480,230,8),noo);
  s+=PT('y',t,'y(t): the prediction',Ln([[480,254],[480,302]],'e',9)+circ(480,320,17,AC.out,9,lab('y','t'))+(oh?chip(vs(m.y[t]),480,352,'middle',9):''),noo);
  s+=PT('loss',t,'loss L(t)',Ln([[498,320],[566,320]],'l',10)+rbox(604,320,72,34,'loss',AC.loss,10)+(oh?chip('L = '+N(m.lt[t]),604,296,'middle',10):''),noo);
  s+=PT('tg',t,'the target',(oh?circ(604,410,17,AC.edge,10,'t')+Ln([[604,392],[604,339]],'e',10)+chip(tLabel(t),604,442,'middle',10):''),noo);
  /* backward overlay */
  if(bk){const cb=gr.contrib.find(c=>c.t===t),dy=oh?gr.dy[t]:null,du=oh?gr.du[t]:null;
    if(oh){s+=Gr([[566,333],[500,333]],0)+gchip('δy = '+vs(dy),534,362,'middle',0);
      s+=Gr([[494,298],[494,256]],1)+gchip('δu = '+vs(du),500,284,'start',1);
      s+=Gr([[494,206],[494,170]],2)+gchip('∇W_hy, ∇b_y ready',500,194,'start',2);}
    const hout=oh?rTv(cfg.Why,du):null,fut=t<cfg.T-1?rTv(cfg.Whh,gr.ds[t+1]):null;
    if(oh)s+=Gr([[494,132],[494,96],[326,96],[320,86]],3);
    if(fut)s+=Gr([[right?722:780,58],[326,58],[316,60]],3);
    if(oh)s+=gchip('from output = '+vs(hout),340,112,'start',3);
    if(fut)s+=gchip('from t+1 = '+vs(fut),520,40,'start',3);
    s+=gchip('δh = '+vs(gr.dh[t]),282,74,'end',3);
    s+=Gr([[314,89],[314,136]],4)+gchip('δs = '+vs(gr.ds[t]),268,158,'end',4);
    s+=Gr([[314,184],[314,233]],5)+Gr([[314,268],[314,332]],5)+Gr([[283,264],[218,264]],5)+gchip('∇W_xh',336,318,'start',5)+gchip('∇W_hh',170,222,'middle',5)+gchip('∇b = δs',372,290,'start',5);
    s+=Gr([[124,264],[left?64:0,264]],6)+gchip('δh(t−1) = '+vs(rTv(cfg.Whh,gr.ds[t])),left?20:60,304,'start',6);}
  return s+'</g>';
}
function cmpRCell(t,ox,left,right){
  const m=RA.m,cfg=S.cfg,ht=m.h[t+1],oh=oHere(t),tt=t+1;let s=`<g transform="translate(${ox},0)">`;
  s+=`<g class="part cmp${RA.t===t?' sel':''}" data-act="expand" data-t="${t}"><title>Click to expand this cell</title><rect class="cellbox" x="50" y="40" width="140" height="350" rx="26"/>`+
   `<text x="120" y="195" text-anchor="middle" font-size="22" font-weight="700" fill="#33414d">RNN</text><text x="120" y="222" text-anchor="middle" font-size="16" fill="#51606c">t = ${tt}</text><text x="120" y="262" text-anchor="middle" font-size="12" fill="#1E5873">▸ click to expand</text></g>`;
  s+=Ln([[left?62:0,250],[50,250]],'h',null,true,3)+(left?circ(45,250,17,AC.h,null,lab('h','t-1'))+chip(vs(m.h[t]),45,284,'middle'):'');
  s+=Ln([[190,70],[right?222:240,70]],'h',null,true,3)+(right?circ(240,70,17,AC.h,null,lab('h','t'))+chip(vs(ht),240,102,'middle'):chip(vs(ht),215,54,'end'));
  s+=Ln([[120,426],[120,390]],'x',null,true,3)+circ(120,445,19,AC.x,null,lab('x','t'))+chip(xLabel(t),150,449,'start');
  if(oh)s+=Ln([[120,40],[120,44]],'o',null,false,3)+circ(120,20,16,AC.out,null,lab('y','t'))+chip(vs(m.y[t]),146,24,'start')+'';
  return s+'</g>';
}
function archRSVG(){
  const cfg=S.cfg,T=cfg.T,times=RA.mode==='cell'?[RA.t]:[...Array(T).keys()];let off=30,body='',prev=null;
  times.forEach((t,n)=>{const exp=RA.mode==='cell'||RA.open.has(t),left=n===0,right=n===times.length-1,w=exp?780:240;
    if(prev){body+=`<path class="ln" d="M${prev} 70 L${prev+30} 70 L${prev+30} 250 L${off+(exp?0:0)} 250" fill="none" stroke="${AC.h}" stroke-width="3" stroke-linejoin="round" marker-end="url(#ah-h)"/>`;}
    body+=exp?expRCell(t,off,left,right,RA.mode==='time'):cmpRCell(t,off,left,right);prev=off+w;off+=w+60;});
  const W=off+10,Hh=490;
  const defs=`<defs>${Object.entries(MK).map(([k,c])=>`<marker id="ah-${k}" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto"><path d="M0,0L10,5L0,10z" fill="${c}"/></marker>`).join('')}</defs>`;
  return{W,H:Hh,html:`${defs}${body}`};
}
/* ---------- explanations of the parts (side panel) ---------- */
function rInfo(nm,t,noHead){
  const cfg=S.cfg,m=RA.m,x=cfg.x[t],hp=m.h[t],ht=m.h[t+1],sv=m.s[t],tt=t+1,AF=ACTS[cfg.act],D=s=>K(s,true),oh=oHere(t),k=outSteps(cfg).indexOf(t);
  const xa=mv(cfg.Wxh,x),ha=mv(cfg.Whh,hp);let title='',what='',tex=[],why='',read='';
  if(nm==='x'){title='Current input';what=`$x_t$ is the new information at step ${tt}${cfg.tok?' (a letter or word written as numbers)':''}.`;tex=[R`x_{${tt}}=${bmc(x)}`];if(cfg.tok)read=`Token: “${tokName(cfg,cfg.tok.seq[t])}”`;}
  else if(nm==='hprev'){title='Memory from the previous step';what=`$h_{t-1}$ is everything the network remembers so far, as a vector.`;tex=[R`h_{${t}}=${bmc(hp)}`];if(t===0)read='At the first step the memory is all zeros.';}
  else if(nm==='mx'){title='Input weights W_xh';what='Each hidden unit has a row of weights saying how much every input number matters to it.';tex=[R`W_{xh}x_t=${bmM(cfg.Wxh)}${bmc(x)}=${bmc(xa)}`];why='Different rows let different hidden units listen to different parts of the input.';}
  else if(nm==='mh'){title='Memory weights W_hh';what='The old memory is mixed by this matrix. It is the loop of the RNN and the same matrix is used at every step.';tex=[R`W_{hh}h_{t-1}=${bmM(cfg.Whh)}${bmc(hp)}=${bmc(ha)}`];why='Sharing W_hh across all steps lets the network handle sequences of any length with a fixed number of weights.';}
  else if(nm==='sum'||nm==='bias'){title=nm==='bias'?'Bias b':'Adding it all up';what=nm==='bias'?'A bias is a number added to each hidden unit. It shifts the result up or down.':'The two products and the bias are added to give the raw new memory $s_t$.';tex=[R`s_t=W_{xh}x_t+W_{hh}h_{t-1}+b`,R`=${bmc(xa)}+${bmc(ha)}+${bmc(cfg.b)}=${bmc(sv)}`];}
  else if(nm==='act'){title=`Activation (${AF.name})`;what=cfg.act==='tanh'?'tanh squeezes every number into the range −1 to 1.':cfg.act==='sigmoid'?'Sigmoid squeezes every number into 0 to 1.':'ReLU keeps positive numbers and turns negative ones into 0.';tex=[R`h_t=${AF.tex}(s_t)=${AF.tex}${bmc(sv)}=${bmc(ht)}`];why='Without an activation the whole network would only be one big matrix multiplication.';}
  else if(nm==='ht'||nm==='hout'){title='New hidden state h(t)';what='The activation output is the new memory. It goes to the next step (where it becomes $h_{t-1}$) and to the output layer.';tex=[R`h_{${tt}}=${bmc(ht)}`];why='The same hidden state is used twice, which is why its gradient later has two parts.';}
  else if(nm==='mo'){title='Output layer W_hy';if(!oh)what='This step has no output (many-to-one: only the last step answers).';else{what='The memory is turned into one raw score per possible answer.';tex=[R`u_t=W_{hy}h_t+b_y=${bmM(cfg.Why)}${bmc(ht)}+${bmc(cfg.by)}=${bmc(m.u[t])}`];}}
  else if(nm==='oact'||nm==='y'){title=cfg.out==='softmax'?'Output function: softmax':cfg.out==='sigmoid'?'Output function: sigmoid':'Output: identity';if(!oh)what='No output at this step.';else{what=cfg.out==='softmax'?'Softmax turns the scores into probabilities that add up to 1.':cfg.out==='sigmoid'?'Sigmoid turns each score into a number between 0 and 1.':'The scores are used as they are.';
    tex=[cfg.out==='softmax'?R`y_c=\dfrac{e^{u_c}}{\sum_q e^{u_q}}`:cfg.out==='sigmoid'?R`y=\sigma(u)`:R`y=u`,R`y_t=${bmc(m.u[t])}\rightarrow${bmc(m.y[t])}`];if(cfg.tok){const y=m.y[t],b=y.indexOf(Math.max(...y));read=`Most likely: “${tokName(cfg,b)}” (${(y[b]*100).toFixed(1)}%)`;}}}
  else if(nm==='loss'||nm==='tg'){title=nm==='tg'?'The target':'Loss';if(!oh)what='No loss at this step.';else{const tg=cfg.t[k];what=nm==='tg'?'The answer the network should have given (the correct one).':cfg.out==='softmax'?'Cross-entropy: only the probability given to the right answer matters.':'Half the squared difference between prediction and target.';
    tex=[cfg.out==='softmax'?R`L_t=-\sum_c t_c\ln y_c=${N(m.lt[t])}`:R`L_t=\tfrac12\sum_c(y_c-t_c)^2=${N(m.lt[t])}`,R`y=${bmc(m.y[t])},\; t=${bmc(tg)}`];}}
  else{title='Cell';what='Click a shape.';}
  return`${noHead?`<h4 class="ih">${rt(title)}</h4>`:`<h3>${rt(title)} <span class="tbadge">t = ${tt}</span></h3>`}<p>${rt(what)}</p>${tex.map(D).join('')}${read?`<div class="read">${esc(read)}</div>`:''}${why?`<p class="why"><b>Why it exists:</b> ${rt(why)}</p>`:''}`;
}
function bInfo(key,t){
  const cfg=S.cfg,m=RA.m,gr=RA.gr,AF=ACTS[cfg.act],D=s=>K(s,true),oh=oHere(t),k=outSteps(cfg).indexOf(t),cb=gr.contrib.find(c=>c.t===t),H=cfg.H;let tex=[],note='';
  const z=Array(H).fill(0),hout=oh?rTv(cfg.Why,gr.du[t]):z,fut=t<cfg.T-1?rTv(cfg.Whh,gr.ds[t+1]):z;
  if(key==='b0'){tex=[R`\delta^y_t=y_t-\text{target}=${bmc(m.y[t])}-${bmc(cfg.t[k])}=${bmc(gr.dy[t])}`];}
  else if(key==='b1'){tex=cfg.out==='sigmoid'?[R`\delta^u_t=\delta^y_t\odot y_t\odot(1-y_t)=${bmc(gr.dy[t])}\odot${bmc(m.y[t])}\odot${bmc(m.y[t].map(v=>1-v))}=${bmc(gr.du[t])}`]:[R`\delta^u_t=\delta^y_t=${bmc(gr.du[t])}`];note=cfg.out==='sigmoid'?'Sigmoid output: multiply by its slope y(1−y).':'For this output function the delta passes through unchanged.';}
  else if(key==='b2'){tex=[R`\nabla W_{hy}=\delta^u_t\,h_t^{\top}=${bmc(gr.du[t])}${bmc(m.h[t+1])}^{\top}=${bmM(cb.cWhy)}`,R`\nabla b_y=\delta^u_t=${bmc(gr.du[t])}`,R`\delta h^{\text{out}}_t=W_{hy}^{\top}\delta^u_t=${bmc(hout)}`];}
  else if(key==='b3'){tex=[R`\delta h_t=\underbrace{W_{hy}^{\top}\delta^u_t}_{\text{from the output}}+\underbrace{W_{hh}^{\top}\delta s_{t+1}}_{\text{from the next step}}`,R`=${bmc(hout)}+${bmc(fut)}=${bmc(gr.dh[t])}`];note=t===cfg.T-1?'This is the last step, so nothing comes from the future yet.':oh?'Both messages are added.':'No output at this step, so only the message from the next step arrives.';}
  else if(key==='b4'){tex=[R`\delta s_t=\delta h_t\odot f'(s_t),\quad f'(s)=${AF.dtex}`,R`=${bmc(gr.dh[t])}\odot${bmc(m.s[t].map((v,j)=>AF.d(v,m.h[t+1][j])))}=${bmc(gr.ds[t])}`];}
  else if(key==='b5'){tex=[R`\nabla W_{xh}=\delta s_t\,x_t^{\top}=${bmM(cb.cWxh)}`,R`\nabla W_{hh}=\delta s_t\,h_{t-1}^{\top}=${bmM(cb.cWhh)}`,R`\nabla b=\delta s_t=${bmc(gr.ds[t])}`];note='These are only this step\'s contributions. The final gradient is the sum over all steps.';}
  else if(key==='b6'){tex=[R`\delta h_{t-1}\;(\text{from this step})=W_{hh}^{\top}\delta s_t=${bmM(cfg.Whh)}^{\top}${bmc(gr.ds[t])}=${bmc(rTv(cfg.Whh,gr.ds[t]))}`];note=t===0?'This is the first step: the error stops here (nothing before it).':'This vector arrives at the previous step as its "from the next step" message.';}
  return tex.map(D).join('')+(note?`<div class="read">${esc(note)}</div>`:'');
}
