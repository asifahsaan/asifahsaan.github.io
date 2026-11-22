function build(){
  const cfg=S.cfg,sh=shapes(cfg),{H,W,k,s,p,q,Hp,Wp,oh,ow,ph,pw,F,C}=sh,n=F*ph*pw,Tg=cfg.target,eta=cfg.eta;
  const {m,g}=backward(cfg),cfg1=applyG(cfg,g,eta),m1=forward(cfg1);
  const st=[],LL=makeLearn(),L_=LL.L,P=o=>st.push(o);
  const mapKeys=(t,f)=>{const a=[],nr=t==='Z'||t==='A'?oh:ph,nc=t==='Z'||t==='A'?ow:pw;for(let i=0;i<nr;i++)for(let j=0;j<nc;j++)a.push(`${t}:${f}:${i}:${j}`);return a;};
  const allMaps=t=>[...Array(F).keys()].flatMap(f=>mapKeys(t,f));
  const vKeys=[...Array(n).keys()].map(x=>`v:${x}`),uKeys=[...Array(C).keys()].map(x=>`u:${x}`),pKeys=[...Array(C).keys()].map(x=>`p:${x}`);
  const filtKeys=f=>{const a=[];for(let i=0;i<k;i++)for(let j=0;j<k;j++)a.push(`W:${f}:${i}:${j}`);return a;};
  const winKeys=(r0,c0,h,w,pre)=>{const a=[];for(let i=0;i<h;i++)for(let j=0;j<w;j++)a.push(`${pre}:${r0+i}:${c0+j}`);return a;};
  const colorCells=(keys,col)=>Object.fromEntries(keys.map(x=>[x,col]));
  const bpL=()=>[...L_('backprop','Backpropagation',`The error starts at the loss and is sent **backwards** through every step of the forward pass, one step at a time, using the chain rule.`,`To improve a weight we need to know how much it contributed to the loss. Backpropagation finds that for every weight with one backward sweep.`),
    ...L_('delta','Error signal $\\delta$',R`$\delta=\dfrac{\partial L}{\partial(\text{value})}$: how much the loss changes if that value changes a little. Red small numbers in the picture are these $\delta$ values.`,`It tells every number which way and how strongly to change.`)];
  // ---- setup
  P({ph:'S',title:'Setup: image, filters and output sizes',path:'X → conv → ReLU → pool → flatten → FC → softmax → L',act:[],gives:[],
    learn:[...L_('cnn','Convolutional neural network (CNN)',`A network for images. Instead of connecting every pixel to every neuron, it slides small **filters** over the image and looks for local patterns (edges, corners, textures).`,`Images have many pixels, so a fully connected network would need a huge number of weights. A filter reuses the same few weights at every position and respects the fact that neighbouring pixels belong together.`),
      ...L_('pixel','Image as numbers',`A grey-scale image is a grid of numbers: 0 = black, 1 = white, values in between are greys.`,`Networks only calculate with numbers, so the image has to be numbers.`),
      ...L_('filter','Filter (kernel)',`A small grid of weights (here ${k}×${k}). It is a pattern detector: it gives a large value where the image looks like the pattern.`,`The filter values are learned. The same filter is used at every position (weight sharing).`),
      ...L_('stride','Stride and padding',`**Stride** is how many pixels the filter moves each time. **Padding** adds a border of zeros around the image.`,`Stride shrinks the output. Padding lets the filter also look at the image border and keeps the output size.`),
      ...L_('notation','Convolution or cross-correlation?',`Deep-learning libraries call this operation convolution, but they do not flip the filter (strictly this is cross-correlation).`,`The flip makes no difference when the filter values are learned, so everyone uses the simpler version.`)],
    what:[`Image $X$ is $${H}\\times${W}$${p?`, zero-padded by $p=${p}$ to $${Hp}\\times${Wp}$`:' (no padding)'}.`,`${F} filter${F>1?'s':''} of size $${k}\\times${k}$ slide with stride $s=${s}$.`,`Then ReLU, $${q}\\times${q}$ max-pool, flatten, a fully connected layer with ${C} classes, softmax and cross-entropy. True class: **${esc(cls(cfg,Tg))}**.`,`Numbers are computed without rounding and shown with 4 decimals. In the picture they are shortened to 2 decimals.`],
    res:[R`o_h=\left\lfloor\frac{H+2p-k}{s}\right\rfloor+1=\left\lfloor\frac{${H}+${2*p}-${k}}{${s}}\right\rfloor+1=${oh},\qquad o_w=\left\lfloor\frac{${W}+${2*p}-${k}}{${s}}\right\rfloor+1=${ow}`,
      R`\text{pool: }\ \lfloor ${oh}/${q}\rfloor\times\lfloor ${ow}/${q}\rfloor=${ph}\times${pw},\qquad n=F\,p_h\,p_w=${F}\cdot${ph}\cdot${pw}=${n}`,
      R`\text{FC shape: }V\in\mathbb R^{${C}\times${n}},\ v\in\mathbb R^{${n}\times1}:\ (${C}\times${n})(${n}\times1)=(${C}\times1)`,
      R`\#\text{parameters}=F(k^2+1)+C(n+1)=${F}(${k*k}+1)+${C}(${n}+1)=${F*(k*k+1)+C*(n+1)}`],
    say:`Before any arithmetic, check the shapes. The size formula tells you how big every map will be. Change image size, stride or padding in the Custom tab and watch these numbers change.`});
  // ---- forward conv
  for(let f=0;f<F;f++)for(let i=0;i<oh;i++)for(let j=0;j<ow;j++){
    const w=cfg.filters[f].w,b=cfg.filters[f].b,r0=i*s,c0=j*s,Wn=m.Xp.slice(r0,r0+k).map(r=>r.slice(c0,c0+k));
    const rows=Wn.map((r,a)=>r.map((x,c)=>ccol(PC.a,Q(x))+ccol(PC.b,Q(w[a][c])))),sum=Wn.reduce((t,r,a)=>t+r.reduce((u,x,c)=>u+x*w[a][c],0),0);
    const win=winKeys(r0,c0,k,k,'X'),zk=`Z:${f}:${i}:${j}`;
    P({ph:'F',title:`Convolve: filter ${f+1} at position (${i+1}, ${j+1})`,path:`window of X (rows ${r0+1}–${r0+k}, cols ${c0+1}–${c0+k}) · W${F>1?`(${f+1})`:''} → Z${F>1?`(${f+1})`:''}[${i+1},${j+1}]`,
      act:[zk],gives:[zk],ov:{boxes:[{keys:win,cls:'w',col:PC.a},{keys:filtKeys(f),cls:'w',col:PC.b}],links:[{from:win,to:zk,cls:'f'}]},
      learn:[...L_('conv','Convolution: the sliding window',`Put the filter on a patch of the image, multiply **pixel by pixel** with the filter weights, add everything up and add the bias. That one number is one cell of the feature map. Then slide the filter and repeat.`,`A large number means "the patch looks like the filter pattern here". Doing it at every position gives a map of where the pattern appears.`),
        ...L_('featmap','Feature map',`The grid of all the numbers produced by one filter, one number per window position.`,`It shows where in the image the filter's pattern was found.`),
        ...L_('bias','Bias $b$',`One extra number per filter, added to every sum.`,`It shifts the result up or down so the filter does not have to respond to a zero patch with exactly zero.`)],
      ops:{cells:{...colorCells(win,PC.a),...colorCells(filtKeys(f),PC.b),[`B:${f}`]:PC.bias},key:[[PC.a,'image window (the patch under the filter)'],[PC.b,'filter weights (the same at every position)'],[PC.bias,'bias']]},
      what:[`Place the $${k}\\times${k}$ filter on the highlighted window (orange), multiply **element by element** with the filter weights (purple), add everything and add the bias (green).`,R`$Z^{(${f+1})}_{${ij(i,j)}}=b_{${f+1}}+\sum_{a,b}W^{(${f+1})}_{ab}\,X_{(${i+1}-1)s+a,\,(${j+1}-1)s+b}$`,`One filter, many positions: the **same** weights are reused everywhere (weight sharing).`],
      res:[R`\text{window}=${ccol(PC.a,bm(Wn))},\quad W^{(${f+1})}=${ccol(PC.b,bm(w))}`,R`\sum_{a,b}W_{ab}X_{ab}=${alignLines(rows.flatMap(r=>r),k)}=${N(sum)}`,R`Z^{(${f+1})}_{${ij(i,j)}}=${N(sum)}+${ccol(PC.bias,Q(b))}=${hl(m.Z[f][i][j])}`],
      say:`This single number says how strongly the window looks like the filter pattern. Press Next to slide the window one step (stride ${s}).`});}
  // ---- relu
  for(let f=0;f<F;f++)P({ph:'F',title:`ReLU on feature map ${f+1}`,path:`Z${F>1?`(${f+1})`:''} → A${F>1?`(${f+1})`:''}`,act:mapKeys('A',f),gives:mapKeys('A',f),
    learn:L_('relu',ACTINFO2.relu.t,ACTINFO2.relu.w,ACTINFO2.relu.y),
    ops:{cells:{...colorCells(mapKeys('Z',f),PC.a)},key:[[PC.a,'values before ReLU (Z)']]},
    what:[R`$A=\max(0,Z)$ is applied to **every** cell of the map.`,`Negative responses become $0$, positive ones pass through unchanged.`],
    res:[R`Z^{(${f+1})}=${bm(m.Z[f])}`,R`A^{(${f+1})}=\max(0,Z^{(${f+1})})=${bm(m.A[f])}`,R`\text{Cell }(1,1)\text{ in detail:}`,...actFormulaLines('relu',m.Z[f][0][0],m.A[f][0][0],R`A_{11}`,R`Z_{11}`)],
    say:`ReLU keeps only the places where the pattern was detected. Dead (zero) cells will pass no gradient backwards later.`});
  // ---- pool
  for(let f=0;f<F;f++)for(let i=0;i<ph;i++)for(let j=0;j<pw;j++){
    const win=winKeys(i*q,j*q,q,q,`A:${f}`),Wn=m.A[f].slice(i*q,i*q+q).map(r=>r.slice(j*q,j*q+q)),[ai,aj]=m.arg[f][i][j],pk=`P:${f}:${i}:${j}`,wk=`A:${f}:${ai}:${aj}`;
    P({ph:'F',title:`Max-pool: window (${i+1}, ${j+1}) of map ${f+1}`,path:`A${F>1?`(${f+1})`:''} window ${q}×${q} → P${F>1?`(${f+1})`:''}[${i+1},${j+1}]`,
      act:[pk],gives:[pk],ov:{boxes:[{keys:win,cls:'w',col:PC.a}],links:[{from:win,to:pk,cls:'f'}]},
      learn:L_('maxpool','Max-pooling',`Cut the map into small non-overlapping windows and keep only the **largest** value of each window.`,`It shrinks the map (less computation) and makes the result tolerant to small shifts: the pattern may move by a pixel and the strongest response is still kept.`),
      ops:{cells:{...colorCells(win,PC.dim),[wk]:PC.a},key:[[PC.a,'the winner: the largest value, it is kept'],[PC.dim,'the other values: dropped']]},
      what:[`Keep only the **largest** value in each non-overlapping $${q}\\times${q}$ window.`,`The winner (orange) is at row ${ai+1}, column ${aj+1} of the map. Remember this position: the gradient will return exactly there.`],
      res:[R`\text{window}=${bm(Wn)}`,R`P^{(${f+1})}_{${ij(i,j)}}=\max(\text{window})=${hl(m.P[f][i][j])}`],
      say:`Pooling shrinks the map ${q}× in each direction and makes the result tolerant to small shifts of the pattern.`});}
  // ---- flatten
  P({ph:'F',title:'Flatten the pooled maps into a vector $v$',path:'P → v',act:[...allMaps('P'),...vKeys],gives:vKeys,
    learn:L_('flatten','Flatten',`Read the 2-D maps row by row and write all numbers into one long list (a vector).`,`The fully connected layer expects a plain list of numbers, not a grid.`),
    what:[`Read the pooled maps row by row (filter by filter) to get a vector of length $n=${n}$.`,`This is only a reshape: no numbers change.`],
    res:[R`v^{\top}=\begin{bmatrix}${m.v.map(N).join(' & ')}\end{bmatrix}`],say:`The fully connected layer needs a plain list of numbers, so the 2-D maps are unrolled.`});
  // ---- fc
  for(let c=0;c<C;c++){const terms=cfg.V[c].map((x,nn)=>ccol(PC.b,Q(x))+ccol(PC.c,Q(m.v[nn])));
    P({ph:'F',title:`Fully connected: logit $u_${c+1}$ (${esc(cls(cfg,c))})`,path:`v, V row ${c+1}, c${c+1} → u${c+1}`,act:[`u:${c}`],gives:[`u:${c}`],edges:`u:${c}`,
      learn:L_('fc','Fully connected layer',`Every flattened number is connected to every class by its own weight. A class score is the weighted sum of all of them plus a bias.`,`After the filters have found patterns, this layer combines them into a decision: which class do these patterns point to?`),
      ops:{cells:colorCells(vKeys,PC.c),key:[[PC.c,'flattened values $v$'],[PC.b,'weights of class '+(c+1)+' (the lines drawn into $u_'+(c+1)+'$)'],[PC.bias,'bias']]},
      what:[R`$u_${c+1}=\sum_{n}V_{${c+1}n}\,v_n+c_${c+1}$`,`Every flattened value (teal) votes for this class with its own weight (purple).`],
      res:[R`u_${c+1}=${alignLines(terms,4)}+${ccol(PC.bias,Q(cfg.c[c]))}=${hl(m.u[c])}`],say:`One dot product per class, exactly like the output layer of the ANN lab.`});}
  // ---- softmax
  const pi=m.p.indexOf(Math.max(...m.p));
  P({ph:'F',title:'Softmax: turn logits into probabilities',path:'u → p',act:[...uKeys,...pKeys],gives:pKeys,
    learn:L_('softmax','Softmax',R`Turns the scores $u_c$ into probabilities $p_c=\dfrac{e^{u_c}}{\sum_j e^{u_j}}$ that are positive and add up to 1.`,`The classes compete: raising one probability lowers the others. The result can be read as "how sure the network is".`),
    what:[R`$p_c=\dfrac{e^{u_c}}{\sum_j e^{u_j}}$`,`Probabilities are positive and sum to 1.`,`Prediction: **${esc(cls(cfg,pi))}** (${(m.p[pi]*100).toFixed(1)}%). The true class **${esc(cls(cfg,Tg))}** gets ${(m.p[Tg]*100).toFixed(1)}%. A random guess would give every class ${(100/C).toFixed(1)}%.`],
    res:[R`\sum_j e^{u_j}=${m.u.map(x=>`e^{${N(x)}}`).join('+')}=${N(m.u.reduce((t,x)=>t+Math.exp(x),0))}`,R`\begin{aligned}${m.p.map((x,c)=>`p_${c+1}&=\\frac{e^{${N(m.u[c])}}}{${N(m.u.reduce((t,y)=>t+Math.exp(y),0))}}=${hl(x)}`).join(R` \\ `)}\end{aligned}`],
    say:`The network currently gives ${N(m.p[Tg])} to the true class (${esc(cls(cfg,Tg))}).`});
  // ---- loss
  P({ph:'F',title:'Cross-entropy loss $L$',path:'p, true class → L',act:['L'],gives:['L'],
    learn:L_('ce','Cross-entropy loss',R`$L=-\sum_c t_c\ln p_c$. With a one-hot target only the true class counts: the loss is $-\ln$ of the probability given to it.`,`Small when the true class gets a high probability, very large when it gets a low one. With softmax it gives the very simple error $p-t$.`),
    ops:{cells:{...Object.fromEntries(pKeys.map((kk,c)=>[kk,c===Tg?PC.loss:PC.dim])),L:PC.loss},key:[[PC.loss,'probability of the **true** class (t = 1): the only one that counts'],[PC.dim,'other classes (t = 0): multiplied by 0, they drop out']]},
    what:[R`$L=-\sum_c t_c\ln p_c$ (note the minus sign: $\ln$ of a probability is negative, so the loss is positive).`,`The true class (${Tg+1}) is outlined in amber, the others in grey: only the amber one is used, because its $t$ is 1 and every other $t$ is 0.`],
    res:[R`L=-\sum_c t_c\ln p_c=-\left[${m.p.map((x,c)=>ccol(c===Tg?PC.loss:'#9aa0a6',c===Tg?R`\mathbf{1\cdot\ln(${N(x)})}`:R`0\cdot\ln(${N(x)})`)).join('+')}\right]`,R`\text{only the true-class term is left: }\ L=-\ln p_{${Tg+1}}=-\ln(${N(m.p[Tg])})=${hl(m.L)}`],say:`Forward pass finished. The backward pass will send this one number back through every layer.`});
  // ---- backward: du
  P({ph:'B',title:'Backward: error at the logits',path:'L → p → u',act:uKeys,grads:uKeys,learn:bpL(),
    what:[R`Softmax + cross-entropy give a very simple result: $\delta^u_c=\dfrac{\partial L}{\partial u_c}=p_c-t_c$.`,`Wrong class too confident → positive error; true class too low → negative error.`],
    res:[R`\begin{aligned}${g.du.map((x,c)=>`\\delta^u_${c+1}&=p_${c+1}-t_${c+1}=${Q(m.p[c])}-${c===Tg?1:0}=${hl(x)}`).join(R` \\ `)}\end{aligned}`],say:`The errors add up to zero: probability taken from one class is given to another.`});
  // ---- backward: dV per class
  for(let c=0;c<C;c++){const items=m.v.map((x,nn)=>R`\mathrm{d}V_{${c+1},${nn+1}}=${ccol(PC.g,Q(g.du[c]))}${ccol(PC.c,Q(x))}=${N(g.dV[c][nn])}`);
    P({ph:'B',title:`Gradient: FC weights of class ${c+1} (row $V_{${c+1}\\cdot}$)`,path:`δu${c+1}, v → dV row ${c+1}, dc${c+1}`,act:[`u:${c}`],edges:`u:${c}`,
      ops:{cells:{[`u:${c}`]:PC.g,...colorCells(vKeys,PC.c)},key:[[PC.g,`error $\\delta^u_${c+1}$ at the class score`],[PC.c,'flattened value $v_n$ (the input of the weight)']]},
      what:[R`$\dfrac{\partial L}{\partial V_{${c+1}n}}=\delta^u_${c+1}\,v_n$ and $\dfrac{\partial L}{\partial c_${c+1}}=\delta^u_${c+1}$`,`Each weight's gradient = (error at the unit it feeds) × (value that went into it).`],
      res:[R`\small ${colLines(items,2)}`,R`\mathrm{d}c_${c+1}=\delta^u_${c+1}=${hl(g.dc[c])}`],say:`Weights attached to large (active) flattened values get the largest gradients.`});}
  // ---- backward: dv
  P({ph:'B',title:'Backward: error arriving at the flattened vector $v$',path:'δu, V → δv',act:vKeys,grads:vKeys,edges:'all',
    ops:{cells:colorCells(uKeys,PC.g),key:[[PC.g,'errors at all class scores, sent back through the FC weights']]},
    what:[R`$\delta^v_n=\sum_c V_{cn}\,\delta^u_c$`,`Each flattened value receives error from **all** classes, weighted by its FC weights.`],
    res:[R`\small ${colLines(m.v.map((_,nn)=>R`\delta^v_{${nn+1}}=${cfg.V.map((row,c)=>ccol(PC.g,`${Q(row[nn])}${Q(g.du[c])}`)).join('+')}=${N(g.dv[nn])}`),1)}`],say:`This is the same hidden-error step as in the ANN lab, now at the flatten layer.`});
  // ---- backward: reshape
  P({ph:'B',title:'Backward: un-flatten into the pooled maps',path:'δv → δP',act:allMaps('P'),grads:allMaps('P'),
    what:[`The reshape is undone: the same numbers are placed back into $${F}$ map${F>1?'s':''} of size $${ph}\\times${pw}$.`],
    res:[g.dP.map((M,f)=>R`\delta^{P(${f+1})}=${bm(M)}`).join(R`,\quad `)],say:`No arithmetic here, only bookkeeping of where each number belongs.`});
  // ---- backward: unpool
  P({ph:'B',title:'Backward: route the error through max-pool',path:'δP → δA (only at the winners)',act:allMaps('A'),grads:allMaps('A'),
    learn:L_('unpool','Error through max-pool',`Max-pool passed only the winner forward, so it sends the error back **only to the winner** of each window. All other cells get 0.`,`Only the winner influenced the output, so only the winner is to blame. Pooling has no weights, yet it decides which cells can learn.`),
    ops:{cells:colorCells(m.arg.flatMap((Af,f)=>Af.flatMap(r=>r.map(([a,b])=>`A:${f}:${a}:${b}`))),PC.a),key:[[PC.a,'the winner cells from the forward pass: they receive the error']]},
    what:[`Max-pool passed only the **winner** forward, so it passes the error back **only to the winner**. Every other cell gets $0$.`,`Winner positions were remembered in the forward pass.`],
    res:[...[...Array(F).keys()].flatMap(f=>[R`\small\text{map }${f+1}:\ ${m.arg[f].flatMap((r,i)=>r.map(([a,b],j)=>`\\delta^P_{${ij(i,j)}}\\to A_{${a+1},${b+1}}`)).join(',\\ ')}`]),g.dA.map((M,f)=>R`\delta^{A(${f+1})}=${bm(M)}`).join(R`,\quad `)],
    say:`This is why pooling has no weights to learn, yet still shapes which cells learn.`});
  // ---- backward: relu
  for(let f=0;f<F;f++)P({ph:'B',title:`Backward: through ReLU (map ${f+1})`,path:`δA${F>1?`(${f+1})`:''} → δZ${F>1?`(${f+1})`:''}`,act:mapKeys('Z',f),grads:mapKeys('Z',f),
    learn:L_('slope','Activation slope $f\'(\\cdot)$',R`How steep the activation is at the current value. For ReLU the slope is 1 where $Z>0$ and 0 where $Z\le 0$.`,`The error must pass through the activation on its way back, so it is multiplied by this slope. A cell that was switched off passes no error.`),
    what:[R`$\delta^Z=\delta^A\odot\mathbb 1[Z>0]$: the gradient passes where $Z>0$ and is blocked where ReLU output zero.`],
    res:[R`\mathbb 1[Z^{(${f+1})}>0]=${bmi(m.Z[f].map(r=>r.map(x=>x>0?1:0)))}`,R`\delta^{Z(${f+1})}=${bm(g.dZ[f])}`],say:`Cells with $Z\\le 0$ were switched off in the forward pass, so they learn nothing from this example.`});
  // ---- backward: dW
  for(let f=0;f<F;f++){for(let a=0;a<k;a++)for(let b=0;b<k;b++){
      const terms=[];const sel=[];for(let i=0;i<oh;i++)for(let j=0;j<ow;j++){sel.push(`X:${i*s+a}:${j*s+b}`);const z=g.dZ[f][i][j];if(z!==0)terms.push(ccol(PC.g,Q(z))+ccol(PC.a,Q(m.Xp[i*s+a][j*s+b])));}
      P({ph:'B',title:`Gradient: filter weight $W^{(${f+1})}_{${a+1}${b+1}}$`,path:`δZ${F>1?`(${f+1})`:''}, X → dW${F>1?`(${f+1})`:''}[${a+1},${b+1}]`,act:[],grads:[`W:${f}:${a}:${b}`],ov:{boxes:sel.map(x=>({keys:[x],cls:'sel'}))},
        learn:L_('sharedgrad','Gradient of a shared weight',`One filter weight was used at **every** position of the image, so its gradient is the **sum** of the contributions from all positions.`,`Each use of the weight influenced the loss, so each use must be counted. The highlighted pixels are exactly the ones this weight multiplied.`),
        ops:{cells:{[`W:${f}:${a}:${b}`]:PC.g,...colorCells(mapKeys('Z',f),PC.g)},key:[[PC.g,'error $\\delta^Z$ at every position (and the weight that receives the gradient)'],[ACC,'blue shaded pixels: the pixels this weight touched']]},
        what:[R`$\dfrac{\partial L}{\partial W_{${a+1}${b+1}}}=\sum_{i,j}\delta^Z_{ij}\,X_{(i-1)s+${a+1},\,(j-1)s+${b+1}}$`,`The **same** weight was used at every position, so its gradient is the **sum** over all positions. Highlighted pixels are the ones this weight touched.`,terms.length<oh*ow?`Terms with $\\delta^Z=0$ (dead ReLU cells) are left out.`:`All positions contribute.`],
        res:[R`\mathrm{d}W^{(${f+1})}_{${a+1}${b+1}}=${terms.length?alignLines(terms,4):'0'}=${hl(g.dW[f][a][b])}`],say:`Weight sharing in action: one number collects evidence from the whole image.`});}
    P({ph:'B',title:`Gradient: bias $b_${f+1}$`,path:`δZ${F>1?`(${f+1})`:''} → db${F>1?`(${f+1})`:''}`,act:[`B:${f}`,...mapKeys('Z',f)],grads:[`B:${f}`],
      what:[R`$\dfrac{\partial L}{\partial b_${f+1}}=\sum_{i,j}\delta^Z_{ij}$ (the bias is added at every position).`],
      res:[R`\mathrm{d}b_${f+1}=${bm(g.dZ[f])}\ \text{summed}=${hl(g.db[f])}`],say:`All gradients are now known.`});}
  // ---- update
  const pl=paramList(cfg),an=analyticGrads(cfg);
  const items=pl.map(pr=>{const v0=getP(cfg,pr.path),v1=getP(cfg1,pr.path);return R`${pr.tex}\leftarrow ${Q(v0)}-${Q(eta)}${Q(an[pr.id])}=${N(v1)}`;});
  P({ph:'U',title:`Update all ${pl.length} parameters`,path:'θ ← θ − η · dθ',act:[],upd:true,
    learn:L_('gd','Gradient descent and learning rate $\\eta$',R`$\theta\leftarrow\theta-\eta\,\dfrac{\partial L}{\partial\theta}$: every parameter takes a small step **against** its gradient. $\eta$ sets the size of the step.`,`That direction reduces the loss fastest. Too large an $\eta$ overshoots, too small is slow.`),
    what:[R`$\theta\leftarrow\theta-\eta\,\dfrac{\partial L}{\partial\theta}$ with $\eta=${N(eta)}$`,`Filters, biases and FC weights all move against their gradient.`],
    res:[R`\small ${colLines(items,2)}`],say:`Gradient descent. The filter cells on the diagram now show the new weights.`});
  P({ph:'C',title:'Check: run the forward pass again',path:'new parameters → p → L′',act:[],upd:true,check:true,gives:[...uKeys,...pKeys,'L'],
    what:[`Same image, **new** parameters.`,R`$L'=${N(m1.L)}$ against $L=${N(m.L)}$`,m1.L<m.L?`**The loss went down**, so the update helped.`:`The loss did **not** go down: try a smaller $\\eta$.`],
    res:[R`p'=(${m1.p.map(N).join(',\\,')}),\quad p_{${Tg+1}}:\ ${N(m.p[Tg])}\to${N(m1.p[Tg])}`,R`L'=-\ln p'_{${Tg+1}}=${hl(m1.L)}\;${m1.L<m.L?'<':'\\ge'}\;${N(m.L)}=L`],
    say:`The true class gained probability. Repeat many times and the filters learn useful patterns (Training tab).`});
  st.gloss=LL.gloss;
  return st;
}
