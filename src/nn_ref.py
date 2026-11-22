"""Python reference for the generic Neural Network Lab (arbitrary layer sizes).
Hidden activation: sigmoid | tanh | relu.  Output+loss: sigmoid+MSE | linear+MSE | softmax+CE.
The JavaScript in neural-network.html must match these numbers to ~1e-12."""
import json, sys
import numpy as np


def default_cfg():
    return dict(sizes=[2, 2, 2], act='sigmoid', out='sigmoid',
                W=[[[.2, .1], [-.3, .2]], [[.3, -.5], [.4, .2]]], b=[[.1, -.2], [0, .1]],
                x=[1, 2], t=[1, 0], eta=.5)


def sig(z): return 1 / (1 + np.exp(-z))


def actf(name, z):
    return sig(z) if name == 'sigmoid' else np.tanh(z) if name == 'tanh' else np.maximum(0, z)


def actd(name, z, a):
    return a * (1 - a) if name == 'sigmoid' else 1 - a * a if name == 'tanh' else (z > 0).astype(float)


def forward(cfg):
    L = len(cfg['sizes']) - 1
    a = [np.array(cfg['x'], float)]
    zs = []
    for l in range(1, L + 1):
        z = np.array(cfg['W'][l - 1], float) @ a[-1] + np.array(cfg['b'][l - 1], float)
        zs.append(z)
        if l < L:
            a.append(actf(cfg['act'], z))
        else:
            if cfg['out'] == 'sigmoid': a.append(sig(z))
            elif cfg['out'] == 'linear': a.append(z.copy())
            else:
                e = np.exp(z - z.max()); a.append(e / e.sum())
    t = np.array(cfg['t'], float)
    y = a[-1]
    loss = -(t * np.log(y)).sum() if cfg['out'] == 'softmax' else .5 * ((y - t) ** 2).sum()
    return dict(a=a, z=zs, L=loss)


def backward(cfg):
    m = forward(cfg)
    L = len(cfg['sizes']) - 1
    t = np.array(cfg['t'], float)
    y = m['a'][-1]
    dy = y - t
    if cfg['out'] == 'sigmoid': dzL = dy * y * (1 - y)
    else: dzL = dy.copy()
    dz = [None] * L
    da = [None] * (L + 1)
    da[L] = dy
    dz[L - 1] = dzL
    for l in range(L - 1, 0, -1):
        da[l] = np.array(cfg['W'][l], float).T @ dz[l]
        dz[l - 1] = da[l] * actd(cfg['act'], m['z'][l - 1], m['a'][l])
    dW = [np.outer(dz[l], m['a'][l]) for l in range(L)]
    db = [dz[l].copy() for l in range(L)]
    return m, dict(da=da, dz=dz, dW=dW, db=db)


def apply(cfg, g, eta):
    c = json.loads(json.dumps(cfg))
    for l in range(len(cfg['W'])):
        c['W'][l] = (np.array(cfg['W'][l]) - eta * g['dW'][l]).tolist()
        c['b'][l] = (np.array(cfg['b'][l]) - eta * g['db'][l]).tolist()
    return c


def numgrad(cfg, eps=1e-6):
    out = {}
    for l in range(len(cfg['W'])):
        for j in range(len(cfg['W'][l])):
            for i in range(len(cfg['W'][l][j])):
                p, q = json.loads(json.dumps(cfg)), json.loads(json.dumps(cfg))
                p['W'][l][j][i] += eps; q['W'][l][j][i] -= eps
                out[f'W{l}_{j}_{i}'] = (forward(p)['L'] - forward(q)['L']) / (2 * eps)
            p, q = json.loads(json.dumps(cfg)), json.loads(json.dumps(cfg))
            p['b'][l][j] += eps; q['b'][l][j] -= eps
            out[f'b{l}_{j}'] = (forward(p)['L'] - forward(q)['L']) / (2 * eps)
    return out


def rand_cfg(sizes, act, out, seed, x, t, eta):
    r = np.random.default_rng(seed)
    W = [np.round(r.uniform(-.8, .8, (sizes[l + 1], sizes[l])), 2).tolist() for l in range(len(sizes) - 1)]
    b = [np.round(r.uniform(-.3, .3, sizes[l + 1]), 2).tolist() for l in range(len(sizes) - 1)]
    return dict(sizes=sizes, act=act, out=out, W=W, b=b, x=x, t=t, eta=eta)


def pack(cfg, epochs=300):
    m, g = backward(cfg)
    c2 = apply(cfg, g, cfg['eta'])
    m2 = forward(c2)
    ng = numgrad(cfg)
    an = {}
    for l in range(len(cfg['W'])):
        for j in range(len(cfg['W'][l])):
            for i in range(len(cfg['W'][l][j])): an[f'W{l}_{j}_{i}'] = g['dW'][l][j, i]
            an[f'b{l}_{j}'] = g['db'][l][j]
    cur, hist = cfg, [m['L']]
    for _ in range(epochs):
        _, gg = backward(cur); cur = apply(cur, gg, cfg['eta']); hist.append(forward(cur)['L'])
    J = lambda v: [x.tolist() if hasattr(x, 'tolist') else x for x in v]
    return dict(cfg=cfg, a=J(m['a']), z=J(m['z']), L=m['L'], da=J(g['da'][1:]), dz=J(g['dz']), dW=J(g['dW']), db=J(g['db']),
                after_L=m2['L'], analytic=an, numeric=ng, maxdiff=max(abs(an[k] - ng[k]) for k in an), hist=hist)


if __name__ == '__main__':
    cfgs = {'lecture': default_cfg(),
            'deep_tanh_linear': rand_cfg([3, 4, 2, 1], 'tanh', 'linear', 3, [.5, -1, .8], [.7], .1),
            'softmax_relu': rand_cfg([2, 3, 3], 'relu', 'softmax', 5, [1.2, -.4], [0, 1, 0], .2),
            'wide_sigmoid': rand_cfg([4, 5, 5, 3], 'sigmoid', 'sigmoid', 9, [.2, .9, -.3, .5], [1, 0, 1], .5)}
    out = {k: pack(v, 1000 if k == 'lecture' else 300) for k, v in cfgs.items()}
    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else 'nn_ref.json', 'w'))
    for k, v in out.items():
        print(k, 'L=%.6f' % v['L'], "L'=%.6f" % v['after_L'], 'maxdiff=%.2e' % v['maxdiff'], 'hist_end=%.5f' % v['hist'][-1])
    d = out['lecture']
    print('lecture z1', np.round(d['z'][0], 4), 'y', np.round(d['a'][2], 4), 'dz1', np.round(d['dz'][0], 4), 'hist[10,100,1000]', [round(d['hist'][i], 5) for i in (1, 10, 100, 1000)])
