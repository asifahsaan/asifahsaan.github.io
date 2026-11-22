function build(){
  const cfg=S.cfg,sz=cfg.sizes,L=sz.length-1;NL=L;const A=ACTS[cfg.act],eta=cfg.eta;
  const {m,g}=backward(cfg),cfg1=apply(cfg,g,eta),m1=forward(cfg1),st=[],LL=makeLearn(),L_=LL.L,P=o=>st.push(o);
  const nk=(l,j)=>`n:${l}:${j}`,ek=(l,j,i)=>`e:${l}:${j}:${i}`;
  const layerNodes=l=>[...Array(sz[l]).keys()].map(j=>nk(l,j));
  const layerEdges=l=>[...Array(sz[l]).keys()].flatMap(j=>[...Array(sz[l-1]).keys()].map(i=>ek(l,j,i)));
  const nodeCols=(keys,col)=>Object.fromEntries(keys.map(k=>[k,col]));
  const actTex=(l)=>l<L?A.tex:(cfg.out==='sigmoid'?R`\sigma`:cfg.out==='linear'?'':R`\mathrm{softmax}`);
  const layerMode=cfg.gran==='layer';
  const soft=cfg.out==='softmax',tgt=cfg.t;
  const plainN=s=>s.replace(/[_{}^()\\]/g,'');
  const outLearn=()=>[...(cfg.out==='sigmoid'?L_('sigout','Sigmoid output',R`$y=\sigma(u)$ squashes each output to a number between 0 and 1.`,`Targets of 0 or 1 live in exactly this range.`):cfg.out==='linear'?L_('linout','Linear output',`The output is just the weighted sum $u$: no squashing.`,`Used when we predict a quantity (price, temperature) rather than a probability.`):L_('softmax','Softmax',R`Turns the scores $u_k$ into probabilities $y_k=\dfrac{e^{u_k}}{\sum_j e^{u_j}}$ that are positive and add up to 1.`,`When exactly one class is correct, the classes compete: raising one probability lowers the others.`))];
  const bpL=()=>[...L_('backprop','Backpropagation',`The error starts at the loss and is sent **backwards**, layer by layer, using the chain rule. Each weight learns how much it contributed to the error.`,`To improve a weight we must know how it changed the loss. Backpropagation computes that for all weights with one backward sweep.`),
    ...L_('delta','Error signal $\\delta$',R`$\delta=\dfrac{\partial L}{\partial(\text{value})}$: how much the loss changes if that value changes a little. Red numbers in the picture are $\delta$ values.`,`It tells every number which way and how strongly to change.`)];
  const slopeL=()=>L_('slope','Activation slope $f\'(\\cdot)$',R`How steep the activation curve is at the current value: sigmoid $a(1-a)$, tanh $1-a^2$, ReLU 1 or 0.`,`The error has to pass through the activation on its way back, so it is multiplied by this slope. Where the curve is flat the error is cut down (vanishing gradient).`);
  // ---- setup
  P({ph:'S',title:`Setup: a ${L===1?'single-layer':L+'-layer'} network with sizes ${sz.join('–')}`,path:'x → hidden layer(s) → output → loss',act:{n:[],e:[]},gives:[],
    learn:[...L_('nn','Neural network',`A stack of layers of simple units (neurons). Each neuron takes the numbers from the previous layer, mixes them with weights, adds a bias and squashes the result with an activation function.`,`Many simple units together can represent very complicated relationships, and the weights can be learned from examples.`),
      ...L_('neuron','Neuron',R`A neuron computes $z=\sum_i w_i a_i+b$ (a weighted sum of its inputs plus a bias) and then outputs $a=f(z)$.`,`It is the basic building block. The weights say how much each input matters.`),
      ...L_('layer','Input, hidden and output layer',`The **input** layer holds the data. **Hidden** layers are in between and learn useful intermediate features. The **output** layer gives the answer.`,`Hidden layers let the network build complex features from simple ones.`),
      ...L_('notation','Different names for the same thing',R`This lab writes weights as $w,v$ (one hidden layer) or $W^{(l)}$ (more layers), pre-activations as $z,u$ and activations as $h,y$. Other books use $\theta$, $a$ or $\hat y$.`,`Only the names differ, the maths is identical.`)],
    what:[`The network has layers of sizes **${sz.join(' → ')}** (input, ${L>1?'hidden, ':''}output). Hidden activation: **${A.name}**. Output: **${cfg.out}**, loss: **${soft?'cross-entropy':'squared error'}**.`,`Step order: forward pass (compute every neuron), loss, backward pass (error signals and gradients), update, check.`,`Numbers are computed without rounding and shown with 4 decimals.`],
    res:[...Array.from({length:L},(_,l)=>R`W^{(${l+1})}:\ (${sz[l+1]}\times${sz[l]}),\quad b^{(${l+1})}:\ (${sz[l+1]}\times1),\quad z^{(${l+1})}=W^{(${l+1})}a^{(${l})}+b^{(${l+1})}:\ (${sz[l+1]}\times${sz[l]})(${sz[l]}\times1)=(${sz[l+1]}\times1)`),
      R`\#\text{parameters}=${Array.from({length:L},(_,l)=>`${sz[l+1]}\\cdot${sz[l]}+${sz[l+1]}`).join('+')}=${paramList(cfg).length}`],
    say:`Before any arithmetic, look at the shapes. Each layer's weight matrix has one row per neuron of that layer and one column per neuron of the previous layer.`});
  // ---- forward
  for(let l=1;l<=L;l++){
    const wLearn=[...L_('preact','Weighted sum $z$ (pre-activation)',`Each neuron multiplies every input by its own weight, adds the products and adds a bias. This number is $z$.`,`It combines everything the neuron sees into one number. It is kept separate from the activation because the backward pass needs both.`),
      ...L_('weight','Weights $w$ and bias $b$',`A weight says how strongly one input pushes a neuron up (positive) or down (negative). The bias is a constant added to the sum.`,`These are the numbers the network learns. The bias lets a neuron switch on even when all inputs are zero.`)];
    if(layerMode){
      P({ph:'F',title:`Forward: all $${tZv(l)}$ of layer ${l} (${lname(l).toLowerCase()})`,path:`${tAv(l-1)}, ${tWm(l)}, ${tBv(l)} → ${tZv(l)}`,act:{n:layerNodes(l),e:layerEdges(l)},learn:wLearn,
        ops:{nodes:nodeCols(layerNodes(l-1),PC.a),edges:Object.fromEntries(layerEdges(l).map(e=>[e,PC.a])),key:[[PC.a,'values of the previous layer and the weights that carry them'],[PC.bias,'bias (above each neuron)']]},
        what:[R`$${tZv(l)}=${tWm(l)}\,${tAv(l-1)}+${tBv(l)}$ (matrix times vector, plus bias).`,`Row $j$ of the matrix is the weights of neuron $j$.`],
        res:[R`${tZv(l)}=${ccol(PC.a,bmM(cfg.W[l-1])+bmc(m.a[l-1]))}+${ccol(PC.bias,bmc(cfg.b[l-1]))}=${bmc(m.z[l-1])}`],
        gives:m.z[l-1].map((v,j)=>({k:`z:${l}:${j}`})),say:`Layer ${l} computes all its weighted sums at once.`});
      if(!(l===L&&cfg.out==='linear'))P({ph:'F',title:l<L?`Forward: activation of layer ${l}`:`Forward: output $y$`,path:`${tZv(l)} → ${tAv(l)}`,act:{n:layerNodes(l),e:[]},
        learn:l<L?L_('act',ACTINFO2[cfg.act].t,ACTINFO2[cfg.act].w,ACTINFO2[cfg.act].y):outLearn(),
        what:[l<L?R`$${tAv(l)}=${A.tex}(${tZv(l)})$ applied to every neuron.`:cfg.out==='softmax'?R`$y_k=\dfrac{e^{u_k}}{\sum_j e^{u_j}}$ (softmax).`:R`$y=\sigma(${tZv(l)})$ (sigmoid output).`],
        res:[R`${tAv(l)}=${actTex(l)}\!\left(${bmc(m.z[l-1])}\right)=${bmc(m.a[l])}`,...(l<L||cfg.out==='sigmoid'?[R`\text{Neuron 1 in detail:}`,...actFormulaLines(l<L?cfg.act:'sigmoid',m.z[l-1][0],m.a[l][0],tA(l,0),tZ(l,0))]:[])],gives:m.a[l].map((v,j)=>({k:`a:${l}:${j}`})),say:`Same activation for every neuron of the layer.`});
      else P({ph:'F',title:'Forward: linear output $y=z$',path:`${tZv(l)} → y`,act:{n:layerNodes(l),e:[]},learn:outLearn(),what:['Linear output: no squashing, $y=z$.'],res:[R`y=${bmc(m.a[l])}`],gives:m.a[l].map((v,j)=>({k:`a:${l}:${j}`})),say:'Useful for regression.'});
    }else{
      for(let j=0;j<sz[l];j++){
        const terms=cfg.W[l-1][j].map((w,i)=>ccol(PC.a,`${Q(w)}${Q(m.a[l-1][i])}`)),srcs=cfg.W[l-1][j].map((_,i)=>tW(l,j,i)+tA(l-1,i)).join('+');
        P({ph:'F',title:`Forward: $${tZ(l,j)}$ of ${lname(l).toLowerCase()} neuron ${j+1}`,path:`${[...Array(sz[l-1]).keys()].map(i=>plainN(tA(l-1,i))).join(', ')}, weights, bias → ${plainN(tZ(l,j))}`,
          act:{n:[nk(l,j)],e:[...Array(sz[l-1]).keys()].map(i=>ek(l,j,i))},learn:wLearn,
          ops:{nodes:nodeCols(layerNodes(l-1),PC.a),edges:Object.fromEntries([...Array(sz[l-1]).keys()].map(i=>[ek(l,j,i),PC.a])),key:[[PC.a,`the ${sz[l-1]} value${sz[l-1]>1?'s':''} from the previous layer, each multiplied by its own weight (the coloured lines)`],[PC.bias,'bias of this neuron (shown above it as b)']]},
          what:[`Neuron ${j+1} of this layer sees **all** ${sz[l-1]} value${sz[l-1]>1?'s':''} from the previous layer (orange).`,R`$${tZ(l,j)}=\sum_i W_{${j+1}i}\,a_i+${tB(l,j)}$`,`One row of the weight matrix times the input vector, plus the bias.`],
          res:[R`${tZ(l,j)}=${srcs}+${tB(l,j)}`,R`=${terms.length>3?alignLines(terms,3)+'+':terms.join('+')+'+'}${ccol(PC.bias,Q(cfg.b[l-1][j]))}=${hl(m.z[l-1][j])}`],gives:[{k:`z:${l}:${j}`}],
          say:`Weighted sum plus bias: the same dot product as a single neuron.`});
        if(!(l===L&&(cfg.out==='softmax'||cfg.out==='linear'))){
          const zz=m.z[l-1][j],isOut=l===L;
          P({ph:'F',title:isOut?`Forward: prediction $y_${j+1}$`:`Forward: activation $${tA(l,j)}$`,path:`${plainN(tZ(l,j))} → ${plainN(tA(l,j))}`,act:{n:[nk(l,j)],e:[]},
            learn:isOut?outLearn():L_('act',ACTINFO2[cfg.act].t,ACTINFO2[cfg.act].w,ACTINFO2[cfg.act].y),
            what:[isOut?R`$y_${j+1}=\sigma(${tZ(l,j)})$ (sigmoid output). Compare with the target $t_${j+1}=${N(cfg.t[j])}$ next.`:R`Squash $${tZ(l,j)}$ with ${A.name}: $${tA(l,j)}=${A.tex}(${tZ(l,j)})$.`,cfg.act==='relu'&&!isOut?`ReLU keeps positive values and sets negative ones to 0.`:`The output of this neuron feeds the next layer.`],
            res:actFormulaLines(isOut?'sigmoid':cfg.act,zz,m.a[l][j],tA(l,j),tZ(l,j)),gives:[{k:`a:${l}:${j}`}],say:isOut?`Target is ${N(cfg.t[j])}, prediction is ${N(m.a[l][j])}.`:`The nonlinearity is what lets a network model more than a straight line.`});
        }else if(l===L&&cfg.out==='linear'){
          P({ph:'F',title:`Forward: linear output $y_${j+1}$`,path:`${plainN(tZ(l,j))} → y`,act:{n:[nk(l,j)],e:[]},learn:outLearn(),what:[R`Linear output: $y_${j+1}=${tZ(l,j)}$, no squashing (regression).`],res:[R`y_${j+1}=${tZ(l,j)}=${hl(m.a[l][j])}`],gives:[{k:`a:${l}:${j}`}],say:`Target is ${N(cfg.t[j])}, prediction is ${N(m.a[l][j])}.`});
        }
      }
      if(l===L&&cfg.out==='softmax'){const pi=m.a[L].indexOf(Math.max(...m.a[L])),ti=cfg.t.indexOf(Math.max(...cfg.t));
        P({ph:'F',title:'Forward: softmax over all outputs',path:'u → y',act:{n:layerNodes(L),e:[]},learn:outLearn(),
          what:[R`$y_k=\dfrac{e^{u_k}}{\sum_j e^{u_j}}$: positive numbers that sum to 1.`,`Prediction: output **${pi+1}** (${(m.a[L][pi]*100).toFixed(1)}%). The target output **${ti+1}** gets ${(m.a[L][ti]*100).toFixed(1)}%. A random guess would give each ${(100/sz[L]).toFixed(1)}%.`],
          res:[R`\sum_j e^{u_j}=${m.z[L-1].map(v=>`e^{${N(v)}}`).join('+')}=${N(m.z[L-1].reduce((s,v)=>s+Math.exp(v),0))}`,R`\begin{aligned}${m.a[L].map((v,k)=>`y_${k+1}&=\\frac{e^{${N(m.z[L-1][k])}}}{${N(m.z[L-1].reduce((s,q)=>s+Math.exp(q),0))}}=${hl(v)}`).join(R` \\ `)}\end{aligned}`],gives:m.a[L].map((v,k)=>({k:`a:${L}:${k}`})),say:`Class probabilities. The target is one-hot.`});}
    }
  }
  // ---- loss
  const lossNodes=[...layerNodes(L),'n:loss'],lossEdges=[...Array(sz[L]).keys()].map(j=>`eL:${j}`);
  const lossOps=()=>({nodes:{...Object.fromEntries(layerNodes(L).map((id,k)=>[id,soft?(tgt[k]>0?PC.loss:PC.dim):PC.loss])),'n:loss':PC.loss},edges:Object.fromEntries(lossEdges.map((e,k)=>[e,soft?(tgt[k]>0?PC.loss:PC.dim):PC.loss])),
    key:soft?[[PC.loss,'probability of the **target** output ($t=1$): the only one that counts'],[PC.dim,'other outputs ($t=0$): multiplied by 0, they drop out']]:[[PC.loss,'each prediction $y$ is compared with its target $t$']]});
  if(soft)P({ph:'F',title:'Forward: cross-entropy loss $L$',path:'y, t → L',act:{n:lossNodes,e:lossEdges},ops:lossOps(),
    learn:L_('ce','Cross-entropy loss',R`$L=-\sum_k t_k\ln y_k$. With a one-hot target only the true class counts: the loss is $-\ln$ of the probability given to it.`,`Small when the true class gets a high probability, very large when it gets a low one. With softmax it gives the simple error $y-t$.`),
    what:[R`$L=-\sum_k t_k\ln y_k$ (note the minus sign: $\ln$ of a probability is negative, so the loss is positive).`,`The target output is outlined in amber, the others in grey: only the amber one is used.`],
    res:[R`L=-\sum_k t_k\ln y_k`,R`=-\left[${m.a[L].map((v,k)=>ccol(cfg.t[k]>0?PC.loss:'#9aa0a6',cfg.t[k]>0?R`\mathbf{${N(cfg.t[k])}\cdot\ln(${N(v)})}`:R`${N(cfg.t[k])}\cdot\ln(${N(v)})`)).join('+')}\right]=${hl(m.L)}`],gives:[{k:'L'}],say:`Forward pass finished. Every backward step reuses the numbers already shown.`});
  else P({ph:'F',title:'Forward: squared-error loss $L$',path:'y, t → L',act:{n:lossNodes,e:lossEdges},ops:lossOps(),
    learn:L_('mse','Squared-error loss',R`$L=\tfrac12\sum_k (y_k-t_k)^2$: the squared gap between each prediction $y$ and its target $t$.`,`Training needs one number that says how wrong the network is. Squaring makes errors positive and punishes big mistakes more. The $\tfrac12$ only tidies the derivative.`),
    what:[R`$L=\tfrac12\sum_k (y_k-t_k)^2$`,`Every prediction $y$ (amber) is compared with its own target $t$.`],
    res:[R`L=\tfrac12\sum_k (y_k-t_k)^2=\tfrac12\left[${m.a[L].map((v,k)=>ccol(PC.loss,`(${N(v)}-${N(cfg.t[k])})^2`)).join('+')}\right]=${hl(m.L)}`],gives:[{k:'L'}],say:`Forward pass finished. Every backward step reuses the numbers already shown.`});
  // ---- backward
  const gradsOfLayer=(l)=>{
    if(layerMode){
      P({ph:'B',title:`Gradient: weights and biases of layer ${l}`,path:`${tDZv(l)}, ${tAv(l-1)} → d${tWm(l)}, d${tBv(l)}`,act:{n:[...layerNodes(l-1),...layerNodes(l)],e:layerEdges(l)},
        gradsOut:{w:[...Array(sz[l]).keys()].flatMap(j=>[...Array(sz[l-1]).keys()].map(i=>`gw:${l}:${j}:${i}`)),b:[...Array(sz[l]).keys()].map(j=>`gb:${l}:${j}`)},
        learn:L_('wgrad','Gradient of a weight',`For one weight: (error at the neuron it feeds) × (value that went into it). The bias gradient is just the error.`,`A weight on a busy input (large value) and on a neuron with a large error is the one most to blame, so it is changed the most.`),
        ops:{nodes:{...nodeCols(layerNodes(l),PC.g),...nodeCols(layerNodes(l-1),PC.a)},edges:Object.fromEntries(layerEdges(l).map(e=>[e,PC.g])),key:[[PC.g,'neurons that hold the error, and the weights that receive a gradient'],[PC.a,'values that went into the weights']]},
        what:[R`$\mathrm{d}${tWm(l)}=${tDZv(l)}\,(${tAv(l-1)})^{\top}$ (outer product) and $\mathrm{d}${tBv(l)}=${tDZv(l)}$.`,`Entry $(j,i)$ is the error at neuron $j$ times the activation that fed the weight.`],
        res:[R`\mathrm{d}${tWm(l)}=${ccol(PC.g,bmc(g.dz[l-1]))}${ccol(PC.a,R`\begin{bmatrix}${m.a[l-1].map(N).join(' & ')}\end{bmatrix}`)}=${bmM(g.dW[l-1])}`,R`\mathrm{d}${tBv(l)}=${bmc(g.db[l-1])}`],say:`All weights of this layer get their gradients at once.`});
    }else for(let j=0;j<sz[l];j++){
      for(let i=0;i<sz[l-1];i++)P({ph:'B',title:`Gradient: $\\mathrm{d}${tW(l,j,i)}$`,path:`δ${j+1}, ${plainN(tA(l-1,i))} → d${plainN(tW(l,j,i))}`,act:{n:[nk(l,j),nk(l-1,i)],e:[ek(l,j,i)]},gradsOut:{w:[`gw:${l}:${j}:${i}`]},
        learn:L_('wgrad','Gradient of a weight',`For one weight: (error at the neuron it feeds) × (value that went into it). The bias gradient is just the error.`,`A weight on a busy input (large value) and on a neuron with a large error is the one most to blame, so it is changed the most.`),
        ops:{nodes:{[nk(l,j)]:PC.g,[nk(l-1,i)]:PC.a},edges:{[ek(l,j,i)]:PC.g},key:[[PC.g,'error at the neuron this weight feeds (and the weight itself)'],[PC.a,'the value that went into the weight']]},
        what:[R`$\dfrac{\partial L}{\partial ${tW(l,j,i)}}=${tDZ(l,j)}\,${tA(l-1,i)}$`,`Error at the neuron (red) times the activation that fed this weight (orange).`],
        res:[R`\mathrm{d}${tW(l,j,i)}=${tDZ(l,j)}\,${tA(l-1,i)}=${ccol(PC.g,Q(g.dz[l-1][j]))}${ccol(PC.a,Q(m.a[l-1][i]))}=${hl(g.dW[l-1][j][i])}`],say:`One weight, one gradient. A big activation means a big share of the blame.`});
      P({ph:'B',title:`Gradient: $\\mathrm{d}${tB(l,j)}$ (bias)`,path:`δ${j+1} → bias`,act:{n:[nk(l,j)],e:[]},gradsOut:{b:[`gb:${l}:${j}`]},
        what:[R`$\dfrac{\partial L}{\partial ${tB(l,j)}}=${tDZ(l,j)}$`,`A bias is a weight whose input is always 1.`],res:[R`\mathrm{d}${tB(l,j)}=${tDZ(l,j)}\cdot 1=${hl(g.db[l-1][j])}`],say:`Bias gradient equals the error signal itself.`});
    }
  };
  const dyk=k=>R`${Q(m.a[L][k])}-${Q(cfg.t[k])}`;
  if(layerMode){
    if(cfg.out==='sigmoid')P({ph:'B',title:'Backward: error at the output layer',path:'L → y → u',act:{n:[...lossNodes,...layerNodes(L)],e:lossEdges},grads:[...layerNodes(L).flatMap((_,k)=>[`da:${L}:${k}`,`dz:${L}:${k}`])],learn:[...bpL(),...slopeL()],
      what:[R`$\delta^y=y-t$, then $\delta^u=\delta^y\odot y\odot(1-y)$ (sigmoid slope).`],res:[R`\delta^y=${bmc(g.da[L])},\quad\delta^u=${bmc(g.dz[L-1])}`],say:`Start of the chain rule.`});
    else P({ph:'B',title:'Backward: error at the output layer',path:'L → y → z',act:{n:[...lossNodes,...layerNodes(L)],e:lossEdges},grads:layerNodes(L).map((_,k)=>`dz:${L}:${k}`),learn:bpL(),
      what:[cfg.out==='softmax'?R`Softmax + cross-entropy give $\delta^u=y-t$.`:R`Linear output + squared error give $\delta^u=y-t$.`],res:[R`${tDZv(L)}=y-t=${bmc(g.dz[L-1])}`],say:`Prediction minus target. That is all.`});
  }else if(cfg.out==='sigmoid'){
    for(let k=0;k<sz[L];k++){
      P({ph:'B',title:`Backward: error at output $y_${k+1}$`,path:`L → y${k+1}`,act:{n:['n:loss',nk(L,k)],e:[`eL:${k}`]},grads:[`da:${L}:${k}`],learn:bpL(),
        what:[R`$${tDA(L,k)}=\dfrac{\partial L}{\partial y_${k+1}}=y_${k+1}-t_${k+1}$`,`Sign tells the direction: ${m.a[L][k]<cfg.t[k]?'prediction too low':'prediction too high'}.`],
        res:[R`${tDA(L,k)}=y_${k+1}-t_${k+1}=${dyk(k)}=${hl(g.da[L][k])}`],say:`Start of the chain rule. The derivative of ½(y−t)² is simply y−t.`});
      P({ph:'B',title:`Backward: error at $${tZ(L,k)}$ (through the sigmoid)`,path:`y${k+1} → ${plainN(tZ(L,k))}`,act:{n:[nk(L,k)],e:[]},grads:[`dz:${L}:${k}`],learn:slopeL(),
        what:[R`$${tDZ(L,k)}=${tDA(L,k)}\,\sigma'(${tZ(L,k)})$, with $\sigma'=y(1-y)$`],
        res:[R`${tDZ(L,k)}=${tDA(L,k)}\,y_${k+1}(1-y_${k+1})=${Q(g.da[L][k])}${Q(m.a[L][k])}(1-${N(m.a[L][k])})=${hl(g.dz[L-1][k])}`],say:`σ′ is at most 0.25, so errors shrink as they pass through sigmoids: the seed of vanishing gradients.`});}
  }else{
    for(let k=0;k<sz[L];k++)P({ph:'B',title:`Backward: error at $${tZ(L,k)}$`,path:`L → y${k+1} → ${plainN(tZ(L,k))}`,act:{n:['n:loss',nk(L,k)],e:[`eL:${k}`]},grads:[`dz:${L}:${k}`],learn:bpL(),
      what:[cfg.out==='softmax'?R`Softmax + cross-entropy collapse to $${tDZ(L,k)}=y_${k+1}-t_${k+1}$.`:R`Linear output + squared error: $${tDZ(L,k)}=y_${k+1}-t_${k+1}$.`,`No separate activation-slope step is needed.`],
      res:[R`${tDZ(L,k)}=y_${k+1}-t_${k+1}=${dyk(k)}=${hl(g.dz[L-1][k])}`],say:`Prediction minus target: the simplest possible error signal.`});
  }
  gradsOfLayer(L);
  for(let l=L-1;l>=1;l--){
    const hidLearn=L_('hiderr','Error arriving at a hidden neuron',`A hidden neuron feeds **every** neuron of the next layer, so it receives error from all of them. The contributions add: each is (the weight that carried the value) × (the error at the next neuron).`,`Errors travel back along the same weights that carried the signal forward.`);
    if(layerMode){
      P({ph:'B',title:`Backward: error arriving at layer ${l}`,path:`${tDZv(l+1)}, ${tWm(l+1)} → ${tDAv(l)}`,act:{n:[...layerNodes(l+1),...layerNodes(l)],e:layerEdges(l+1)},grads:layerNodes(l).map((_,j)=>`da:${l}:${j}`),learn:hidLearn,
        ops:{nodes:{...nodeCols(layerNodes(l+1),PC.c),...nodeCols(layerNodes(l),PC.g)},edges:Object.fromEntries(layerEdges(l+1).map(e=>[e,PC.c])),key:[[PC.c,'errors of the next layer and the weights they travel back along']]},
        what:[R`$${tDAv(l)}=(${tWm(l+1)})^{\top}${tDZv(l+1)}$: errors travel back along the same weights.`],res:[R`${tDAv(l)}=${bmM(cfg.W[l])}^{\top}${bmc(g.dz[l])}=${bmc(g.da[l])}`],say:`Every neuron gets error from all neurons it feeds; contributions add.`});
      P({ph:'B',title:`Backward: error at the pre-activation of layer ${l}`,path:`${tDAv(l)} → ${tDZv(l)}`,act:{n:layerNodes(l),e:[]},grads:layerNodes(l).map((_,j)=>`dz:${l}:${j}`),learn:slopeL(),
        what:[R`$${tDZv(l)}=${tDAv(l)}\odot f'(${tZv(l)})$ with $f'=${A.dtex}$`],res:[R`${tDZv(l)}=${bmc(g.da[l])}\odot${bmc(m.a[l].map((a,j)=>A.d(m.z[l-1][j],a)))}=${bmc(g.dz[l-1])}`],say:`Multiply by the slope of the activation.`});
    }else for(let j=0;j<sz[l];j++){
      const terms=cfg.W[l].map((row,k)=>ccol(PC.c,`${Q(row[j])}${Q(g.dz[l][k])}`));
      P({ph:'B',title:`Backward: error arriving at $${tA(l,j)}$`,path:`δ of next layer, weights → ${plainN(tA(l,j))}`,act:{n:[nk(l,j)],e:cfg.W[l].map((_,k)=>ek(l+1,k,j))},grads:[`da:${l}:${j}`],learn:hidLearn,
        ops:{nodes:nodeCols(layerNodes(l+1),PC.c),edges:Object.fromEntries(cfg.W[l].map((_,k)=>[ek(l+1,k,j),PC.c])),key:[[PC.c,`errors of the ${sz[l+1]} neuron${sz[l+1]>1?'s':''} of the next layer, each with the weight that connects it to this neuron`]]},
        what:[R`$${tDA(l,j)}=\sum_k W_{kj}\,\delta^{z}_k$ over the ${sz[l+1]} neuron${sz[l+1]>1?'s':''} it feeds`,sz[l+1]>1?`**All** of them send error back (teal), so the contributions add.`:`Only one neuron receives this value, so there is one contribution.`],
        res:[R`${tDA(l,j)}=${cfg.W[l].map((_,k)=>`${tW(l+1,k,j)}${tDZ(l+1,k)}`).join('+')}`,R`=${terms.join('+')}=${hl(g.da[l][j])}`],say:`This extra step (compared with a network without hidden layers) spreads the blame over earlier neurons.`});
      P({ph:'B',title:`Backward: error at $${tZ(l,j)}$ (through ${A.name})`,path:`${plainN(tA(l,j))} → ${plainN(tZ(l,j))}`,act:{n:[nk(l,j)],e:[]},grads:[`dz:${l}:${j}`],learn:slopeL(),
        what:[R`$${tDZ(l,j)}=${tDA(l,j)}\,f'(${tZ(l,j)})$, with $f'=${A.dtex}$`],
        res:[R`${tDZ(l,j)}=${Q(g.da[l][j])}\cdot ${N(A.d(m.z[l-1][j],m.a[l][j]))}=${hl(g.dz[l-1][j])}`],say:cfg.act==='relu'?`ReLU slope is 1 where $z>0$ and 0 elsewhere.`:`Same slope scaling as before.`});
    }
    gradsOfLayer(l);
  }
  // ---- update
  const pl=paramList(cfg),an=analytic(cfg);
  const items=pl.map(pr=>R`${pr.tex}\leftarrow ${Q(getP(cfg,pr.path))}-${Q(eta)}${Q(an[pr.id])}=${N(getP(cfg1,pr.path))}`);
  P({ph:'U',title:`Update all ${pl.length} parameters`,path:'θ ← θ − η · dθ',act:{n:[],e:[]},upd:true,
    learn:L_('gd','Gradient descent and learning rate $\\eta$',R`$\theta\leftarrow\theta-\eta\,\dfrac{\partial L}{\partial\theta}$: every parameter takes a small step **against** its gradient. $\eta$ sets the size of the step.`,`That direction reduces the loss fastest. Too large an $\eta$ overshoots, too small is slow.`),
    what:[R`$\theta\leftarrow\theta-\eta\,\dfrac{\partial L}{\partial\theta}$ with $\eta=${N(eta)}$`,`Step against the gradient, so the loss should go down.`],
    res:[R`\small ${colLines(items,2)}`],say:`Gradient descent. Edge labels on the diagram now show the new weights.`});
  P({ph:'C',title:'Check: run the forward pass again',path:'new weights → y → L′',act:{n:[],e:[]},upd:true,check:true,
    what:[`Same inputs, **new** weights.`,R`$L'=${N(m1.L)}$ against $L=${N(m.L)}$`,m1.L<m.L?`**The loss went down**, so the update helped.`:`The loss did **not** go down: try a smaller $\\eta$.`],
    res:[R`y'=(${m1.a[L].map(N).join(',\\,')}),\quad t=(${cfg.t.map(N).join(',\\,')})`,R`L'=${hl(m1.L)}\;${m1.L<m.L?'<':'\\ge'}\;${N(m.L)}=L`],say:`Repeat this loop many times and you get training (see the Training tab).`});
  st.gloss=LL.gloss;
  return st;
}
