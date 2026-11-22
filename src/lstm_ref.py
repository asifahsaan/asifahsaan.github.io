"""Python reference for the LSTM / GRU Lab (BPTT). The JavaScript in lstm.html must match to ~1e-12.
LSTM: f,i,o = sigmoid(W x + U h + b), g = tanh(...), c_t = f*c_{t-1} + i*g, h_t = o*tanh(c_t)
GRU : z,r = sigmoid(W x + U h + b), n = tanh(Wn x + Un (r*h) + bn), h_t = (1-z)*h_{t-1} + z*n
Read-out: u_t = Why h_t + by, y_t = sigmoid | linear | softmax. Loss summed over output steps."""
import json, sys
import numpy as np

GATES = {'lstm': ['f', 'i', 'g', 'o'], 'gru': ['z', 'r', 'n']}


def sig(z): return 1 / (1 + np.exp(-z))


def rand_cfg(cell, T, d, H, C, out, mode, seed, eta):
    r = np.random.default_rng(seed); R = lambda *s: np.round(r.uniform(-.8, .8, s), 2).tolist()
    p = {k: dict(W=R(H, d), U=R(H, H), b=np.round(r.uniform(-.3, .3, H), 2).tolist()) for k in GATES[cell]}
    if cell == 'lstm': p['f']['b'] = [round(v + .5, 2) for v in p['f']['b']]  # common forget-bias trick
    k = T if mode == 'all' else 1
    t = (np.eye(C)[[j % C for j in range(k)]].tolist() if out == 'softmax' else [[(q + j + 1) % 2 for j in range(C)] for q in range(k)])
    return dict(cell=cell, T=T, d=d, H=H, C=C, out=out, mode=mode, p=p, Why=R(C, H),
                by=np.round(r.uniform(-.2, .2, C), 2).tolist(), x=np.round(r.uniform(-1, 1, (T, d)), 2).tolist(), t=t, eta=eta)


def outs(cfg): return list(range(cfg['T'])) if cfg['mode'] == 'all' else [cfg['T'] - 1]


def pre(P, k, x, h): return np.array(P[k]['W'], float) @ x + np.array(P[k]['U'], float) @ h + np.array(P[k]['b'], float)


def forward(cfg):
    T, H, P = cfg['T'], cfg['H'], cfg['p']
    h = [np.zeros(H)]; c = [np.zeros(H)]; G = []
    for t in range(T):
        x = np.array(cfg['x'][t], float); hp = h[-1]
        if cfg['cell'] == 'lstm':
            f, i, o = sig(pre(P, 'f', x, hp)), sig(pre(P, 'i', x, hp)), sig(pre(P, 'o', x, hp)); g = np.tanh(pre(P, 'g', x, hp))
            ct = f * c[-1] + i * g; ht = o * np.tanh(ct); c.append(ct); h.append(ht); G.append(dict(f=f, i=i, g=g, o=o))
        else:
            z, r = sig(pre(P, 'z', x, hp)), sig(pre(P, 'r', x, hp))
            n = np.tanh(np.array(P['n']['W'], float) @ x + np.array(P['n']['U'], float) @ (r * hp) + np.array(P['n']['b'], float))
            h.append((1 - z) * hp + z * n); G.append(dict(z=z, r=r, n=n))
    Why, by = np.array(cfg['Why'], float), np.array(cfg['by'], float)
    u, y, lt = {}, {}, {}
    for k, t in enumerate(outs(cfg)):
        ut = Why @ h[t + 1] + by; u[t] = ut
        if cfg['out'] == 'sigmoid': y[t] = sig(ut)
        elif cfg['out'] == 'linear': y[t] = ut.copy()
        else:
            e = np.exp(ut - ut.max()); y[t] = e / e.sum()
        tg = np.array(cfg['t'][k], float)
        lt[t] = -(tg * np.log(y[t])).sum() if cfg['out'] == 'softmax' else .5 * ((y[t] - tg) ** 2).sum()
    return dict(h=h, c=c, G=G, u=u, y=y, lt=lt, L=sum(lt.values()))


def backward(cfg):
    m = forward(cfg); T, H, d, C = cfg['T'], cfg['H'], cfg['d'], cfg['C']; P = cfg['p']; lstm = cfg['cell'] == 'lstm'
    Why = np.array(cfg['Why'], float); os_ = outs(cfg)
    gr = {k: dict(W=np.zeros((H, d)), U=np.zeros((H, H)), b=np.zeros(H)) for k in GATES[cfg['cell']]}
    dWhy = np.zeros_like(Why); dby = np.zeros(C)
    dh_next = np.zeros(H); dc_next = np.zeros(H)
    steps = [None] * T
    for t in range(T - 1, -1, -1):
        info = {}
        dh = dh_next.copy()
        if t in os_:
            tg = np.array(cfg['t'][os_.index(t)], float); y = m['y'][t]
            dy = y - tg; du = dy * y * (1 - y) if cfg['out'] == 'sigmoid' else dy.copy()
            dh += Why.T @ du; dWhy += np.outer(du, m['h'][t + 1]); dby += du; info.update(dy=dy, du=du)
        info['dh'] = dh.copy()
        x = np.array(cfg['x'][t], float); hp = m['h'][t]; g = m['G'][t]
        if lstm:
            cp, ct = m['c'][t], m['c'][t + 1]; th = np.tanh(ct)
            do = dh * th; dc = dc_next + dh * g['o'] * (1 - th ** 2)
            df, di, dg = dc * cp, dc * g['g'], dc * g['i']
            dp = dict(f=df * g['f'] * (1 - g['f']), i=di * g['i'] * (1 - g['i']), g=dg * (1 - g['g'] ** 2), o=do * g['o'] * (1 - g['o']))
            dpost = dict(f=df, i=di, g=dg, o=do)
            dh_next = sum(np.array(P[k]['U'], float).T @ dp[k] for k in dp); dc_next = dc * g['f']
            info.update(dc=dc, dpost=dpost, dp=dp, dc_prev=dc_next.copy(), dh_prev=dh_next.copy())
        else:
            z, r, n = g['z'], g['r'], g['n']
            dz = dh * (n - hp); dn = dh * z; dh_prev = dh * (1 - z)
            dpn = dn * (1 - n ** 2); Un = np.array(P['n']['U'], float); drh = Un.T @ dpn
            dr = drh * hp; dh_prev = dh_prev + drh * r
            dpz, dpr = dz * z * (1 - z), dr * r * (1 - r)
            dp = dict(z=dpz, r=dpr, n=dpn); dpost = dict(z=dz, r=dr, n=dn)
            dh_prev = dh_prev + np.array(P['z']['U'], float).T @ dpz + np.array(P['r']['U'], float).T @ dpr
            dh_next = dh_prev; info.update(dpost=dpost, dp=dp, drh=drh, dh_prev=dh_prev.copy())
        for k in dp:
            gr[k]['W'] += np.outer(dp[k], x)
            hin = (g['r'] * hp) if (not lstm and k == 'n') else hp
            gr[k]['U'] += np.outer(dp[k], hin); gr[k]['b'] += dp[k]
        steps[t] = info
    return m, dict(gr=gr, dWhy=dWhy, dby=dby, steps=steps)


def paths(cfg):
    P = []
    for k in GATES[cfg['cell']]:
        for nm in ('W', 'U'):
            A = np.array(cfg['p'][k][nm])
            for idx in np.ndindex(A.shape): P.append((f'{k}.{nm}_' + '_'.join(map(str, idx)), (k, nm) + idx))
        for j in range(cfg['H']): P.append((f'{k}.b_{j}', (k, 'b', j)))
    for nm in ('Why',):
        A = np.array(cfg[nm])
        for idx in np.ndindex(A.shape): P.append((f'Why_' + '_'.join(map(str, idx)), ('Why',) + idx))
    for j in range(cfg['C']): P.append((f'by_{j}', ('by', j)))
    return P


def getp(cfg, p):
    if p[0] in ('Why', 'by'): o = cfg[p[0]]; idx = p[1:]
    else: o = cfg['p'][p[0]][p[1]]; idx = p[2:]
    for i in idx: o = o[i]
    return o


def setp(cfg, p, v):
    if p[0] in ('Why', 'by'): o = cfg[p[0]]; idx = p[1:]
    else: o = cfg['p'][p[0]][p[1]]; idx = p[2:]
    for i in idx[:-1]: o = o[i]
    o[idx[-1]] = v


def agrad(cfg, g, p):
    if p[0] == 'Why': return g['dWhy'][p[1], p[2]]
    if p[0] == 'by': return g['dby'][p[1]]
    A = g['gr'][p[0]][p[1]]; return A[p[2:]] if len(p) > 2 else A


def apply(cfg, g, eta):
    c = json.loads(json.dumps(cfg))
    for _, p in paths(cfg): setp(c, p, getp(cfg, p) - eta * agrad(cfg, g, p))
    return c


def numgrad(cfg, eps=1e-6):
    o = {}
    for key, p in paths(cfg):
        a, b = json.loads(json.dumps(cfg)), json.loads(json.dumps(cfg))
        setp(a, p, getp(cfg, p) + eps); setp(b, p, getp(cfg, p) - eps)
        o[key] = (forward(a)['L'] - forward(b)['L']) / (2 * eps)
    return o


def pack(cfg, epochs=200):
    m, g = backward(cfg); c2 = apply(cfg, g, cfg['eta']); m2 = forward(c2); ng = numgrad(cfg)
    an = {key: float(agrad(cfg, g, p)) for key, p in paths(cfg)}
    cur, hist = cfg, [m['L']]
    for _ in range(epochs):
        _, gg = backward(cur); cur = apply(cur, gg, cfg['eta']); hist.append(forward(cur)['L'])
    L = lambda v: [x.tolist() for x in v]
    return dict(cfg=cfg, h=L(m['h']), c=L(m['c']), G=[{k: v.tolist() for k, v in gg.items()} for gg in m['G']],
                y={str(k): v.tolist() for k, v in m['y'].items()}, L=m['L'], after_L=m2['L'], analytic=an, numeric=ng,
                maxdiff=max(abs(an[k] - ng[k]) for k in an), hist=hist,
                dh=[s['dh'].tolist() for s in g['steps']], dWhy=g['dWhy'].tolist())


if __name__ == '__main__':
    cfgs = {'lstm_default': rand_cfg('lstm', 2, 2, 2, 2, 'sigmoid', 'last', 11, .5),
            'gru_default': rand_cfg('gru', 2, 2, 2, 2, 'sigmoid', 'last', 12, .5),
            'lstm_all_softmax': rand_cfg('lstm', 4, 2, 3, 3, 'softmax', 'all', 3, .2),
            'gru_all_linear': rand_cfg('gru', 4, 3, 3, 1, 'linear', 'all', 4, .1),
            'lstm_long': rand_cfg('lstm', 5, 1, 2, 1, 'sigmoid', 'last', 5, .5)}
    out = {k: pack(v) for k, v in cfgs.items()}
    json.dump(out, open('lstm_ref.json', 'w'))
    for k, v in out.items(): print(k, 'L=%.6f' % v['L'], "L'=%.6f" % v['after_L'], 'maxdiff=%.2e' % v['maxdiff'], 'end=%.5f' % v['hist'][-1])
    # defaults for the JS page (single source of truth)
    D = {'lstm': cfgs['lstm_default'], 'gru': cfgs['gru_default']}
    open('lstm_defaults.js', 'w').write('const DEFAULTS=' + json.dumps(D) + ';\n')
