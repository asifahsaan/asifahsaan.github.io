"""Python reference for the RNN Lab (vanilla RNN, backpropagation through time).
s_t = Wxh x_t + Whh h_{t-1} + b ; h_t = f(s_t) ; h_0 = 0
outputs (at the last step, or at every step): u_t = Why h_t + by ; y_t = g(u_t)
loss: sum over output steps of  sigmoid/linear -> 1/2 sum (y-t)^2 ;  softmax -> -sum t ln y.
The JavaScript in rnn.html must match these numbers to ~1e-12."""
import json, sys
import numpy as np


def default_cfg():
    return dict(T=2, d=2, H=2, C=2, act='tanh', out='sigmoid', mode='last',
                Wxh=[[.5, -.3], [.2, .4]], Whh=[[.1, .3], [-.2, .2]], b=[.1, -.1],
                Why=[[.6, -.4], [.3, .5]], by=[0, .1],
                x=[[1, .5], [.5, -1]], t=[[1, 0]], eta=.5)


def sig(z): return 1 / (1 + np.exp(-z))


def f(name, z): return np.tanh(z) if name == 'tanh' else sig(z) if name == 'sigmoid' else np.maximum(0, z)


def fd(name, z, h): return 1 - h * h if name == 'tanh' else h * (1 - h) if name == 'sigmoid' else (z > 0).astype(float)


def out_steps(cfg): return list(range(cfg['T'])) if cfg['mode'] == 'all' else [cfg['T'] - 1]


def forward(cfg):
    T, H = cfg['T'], cfg['H']
    Wxh, Whh, b = np.array(cfg['Wxh'], float), np.array(cfg['Whh'], float), np.array(cfg['b'], float)
    Why, by = np.array(cfg['Why'], float), np.array(cfg['by'], float)
    h = [np.zeros(H)]; s = []; u = {}; y = {}
    for t in range(T):
        st = Wxh @ np.array(cfg['x'][t], float) + Whh @ h[-1] + b
        s.append(st); h.append(f(cfg['act'], st))
    loss_t = {}
    for k, t in enumerate(out_steps(cfg)):
        ut = Why @ h[t + 1] + by
        u[t] = ut
        if cfg['out'] == 'sigmoid': y[t] = sig(ut)
        elif cfg['out'] == 'linear': y[t] = ut.copy()
        else:
            e = np.exp(ut - ut.max()); y[t] = e / e.sum()
        tg = np.array(cfg['t'][k], float)
        loss_t[t] = -(tg * np.log(y[t])).sum() if cfg['out'] == 'softmax' else .5 * ((y[t] - tg) ** 2).sum()
    return dict(h=h, s=s, u=u, y=y, lt=loss_t, L=sum(loss_t.values()))


def backward(cfg):
    m = forward(cfg)
    T, H = cfg['T'], cfg['H']
    Whh, Why = np.array(cfg['Whh'], float), np.array(cfg['Why'], float)
    outs = out_steps(cfg)
    dh = [np.zeros(H) for _ in range(T)]; ds = [None] * T
    du = {}; dy = {}; dWhy = np.zeros_like(Why); dby = np.zeros(Why.shape[0])
    dWxh = np.zeros((H, cfg['d'])); dWhh = np.zeros((H, H)); db = np.zeros(H)
    contrib = []
    for t in range(T - 1, -1, -1):
        if t in outs:
            k = outs.index(t); tg = np.array(cfg['t'][k], float); y = m['y'][t]
            dy[t] = y - tg
            du[t] = dy[t] * y * (1 - y) if cfg['out'] == 'sigmoid' else dy[t].copy()
            dh[t] = dh[t] + Why.T @ du[t]
            dWhy += np.outer(du[t], m['h'][t + 1]); dby += du[t]
        if t < T - 1: dh[t] = dh[t] + Whh.T @ ds[t + 1]
        ds[t] = dh[t] * fd(cfg['act'], m['s'][t], m['h'][t + 1])
        cWxh = np.outer(ds[t], np.array(cfg['x'][t], float)); cWhh = np.outer(ds[t], m['h'][t])
        dWxh += cWxh; dWhh += cWhh; db += ds[t]
        contrib.append((t, cWxh, cWhh))
    return m, dict(dh=dh, ds=ds, du=du, dy=dy, dWxh=dWxh, dWhh=dWhh, db=db, dWhy=dWhy, dby=dby)


NAMES = ['Wxh', 'Whh', 'b', 'Why', 'by']
GN = dict(Wxh='dWxh', Whh='dWhh', b='db', Why='dWhy', by='dby')


def apply(cfg, g, eta):
    c = json.loads(json.dumps(cfg))
    for n in NAMES: c[n] = (np.array(cfg[n], float) - eta * g[GN[n]]).tolist()
    return c


def numgrad(cfg, eps=1e-6):
    out = {}
    for n in NAMES:
        a = np.array(cfg[n], float)
        for idx in np.ndindex(a.shape):
            p, q = json.loads(json.dumps(cfg)), json.loads(json.dumps(cfg))
            ap, aq = a.copy(), a.copy(); ap[idx] += eps; aq[idx] -= eps
            p[n], q[n] = ap.tolist(), aq.tolist()
            out[n + '_' + '_'.join(map(str, idx))] = (forward(p)['L'] - forward(q)['L']) / (2 * eps)
    return out


def rand_cfg(T, d, H, C, act, out, mode, seed, eta):
    r = np.random.default_rng(seed); R = lambda *s: np.round(r.uniform(-.7, .7, s), 2).tolist()
    k = T if mode == 'all' else 1
    t = (np.eye(C)[r.integers(0, C, k)].tolist() if out == 'softmax' else np.round(r.uniform(0, 1, (k, C)), 1).tolist())
    return dict(T=T, d=d, H=H, C=C, act=act, out=out, mode=mode, Wxh=R(H, d), Whh=R(H, H), b=R(H), Why=R(C, H), by=R(C),
                x=np.round(r.uniform(-1, 1, (T, d)), 2).tolist(), t=t, eta=eta)


def pack(cfg, epochs=300):
    m, g = backward(cfg)
    c2 = apply(cfg, g, cfg['eta']); m2 = forward(c2); ng = numgrad(cfg)
    an = {}
    for n in NAMES:
        a = np.array(g[GN[n]])
        for idx in np.ndindex(a.shape): an[n + '_' + '_'.join(map(str, idx))] = a[idx]
    cur, hist = cfg, [m['L']]
    for _ in range(epochs):
        _, gg = backward(cur); cur = apply(cur, gg, cfg['eta']); hist.append(forward(cur)['L'])
    J = lambda v: [x.tolist() for x in v]
    D = lambda d: {str(k): v.tolist() for k, v in d.items()}
    return dict(cfg=cfg, h=J(m['h']), s=J(m['s']), u=D(m['u']), y=D(m['y']), L=m['L'], dh=J(g['dh']), ds=J(g['ds']), du=D(g['du']),
                dWxh=g['dWxh'].tolist(), dWhh=g['dWhh'].tolist(), db=g['db'].tolist(), dWhy=g['dWhy'].tolist(), dby=g['dby'].tolist(),
                after_L=m2['L'], analytic=an, numeric=ng, maxdiff=max(abs(an[k] - ng[k]) for k in an), hist=hist)


if __name__ == '__main__':
    cfgs = {'lecture': default_cfg(),
            'many_to_many': rand_cfg(4, 2, 3, 2, 'tanh', 'sigmoid', 'all', 1, .3),
            'softmax_relu': rand_cfg(3, 3, 2, 3, 'relu', 'softmax', 'last', 2, .2),
            'sigmoid_linear': rand_cfg(5, 1, 4, 1, 'sigmoid', 'linear', 'all', 3, .1)}
    out = {k: pack(v, 500 if k == 'lecture' else 200) for k, v in cfgs.items()}
    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else 'rnn_ref.json', 'w'))
    for k, v in out.items():
        print(k, 'L=%.6f' % v['L'], "L'=%.6f" % v['after_L'], 'maxdiff=%.2e' % v['maxdiff'], 'hist_end=%.6f' % v['hist'][-1])
    d = out['lecture']; np.set_printoptions(precision=4, suppress=True)
    print('h', np.array(d['h']), 'y', d['y'], 'ds', np.array(d['ds']))
