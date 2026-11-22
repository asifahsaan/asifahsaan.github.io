"""Python reference for the CNN Lab. The JavaScript in cnn.html must match these numbers to ~1e-12.
Pipeline: image -> conv(F filters, k x k, stride s, zero padding p) + bias -> ReLU -> max-pool (q x q, stride q)
          -> flatten -> fully connected (C classes) -> softmax -> cross-entropy."""
import json, sys
import numpy as np

IMG = [
    [.10, .15, .85, .90, .12, .08],
    [.12, .10, .90, .80, .10, .14],
    [.08, .16, .88, .92, .15, .10],
    [.14, .09, .91, .86, .11, .13],
    [.10, .12, .84, .90, .09, .15],
    [.15, .08, .87, .83, .14, .11],
]
VEDGE = [[1, 0, -1], [1, 0, -1], [1, 0, -1]]


def default_cfg():
    return dict(img=IMG, filters=[dict(w=VEDGE, b=0.1)], stride=1, pad=0, q=2,
                V=[[-.3, .2, .6, -.1], [.5, .3, -.2, .4]], c=[0.0, 0.1], target=0, eta=.1)


def shapes(cfg):
    H, W = np.array(cfg['img']).shape
    k = len(cfg['filters'][0]['w'])
    s, p, q = cfg['stride'], cfg['pad'], cfg['q']
    oh, ow = (H + 2 * p - k) // s + 1, (W + 2 * p - k) // s + 1
    return H, W, k, oh, ow, oh // q, ow // q


def forward(cfg, th=None):
    th = th or cfg
    X = np.array(cfg['img'], float)
    H, W, k, oh, ow, ph, pw = shapes(cfg)
    s, p, q = cfg['stride'], cfg['pad'], cfg['q']
    Xp = np.pad(X, p)
    F = len(th['filters'])
    Z = np.zeros((F, oh, ow))
    for f in range(F):
        Wf = np.array(th['filters'][f]['w'], float)
        for i in range(oh):
            for j in range(ow):
                Z[f, i, j] = (Wf * Xp[i * s:i * s + k, j * s:j * s + k]).sum() + th['filters'][f]['b']
    A = np.maximum(Z, 0)
    P = np.zeros((F, ph, pw))
    arg = np.zeros((F, ph, pw, 2), int)
    for f in range(F):
        for i in range(ph):
            for j in range(pw):
                win = A[f, i * q:i * q + q, j * q:j * q + q]
                a = np.argmax(win)  # first maximum
                arg[f, i, j] = (i * q + a // q, j * q + a % q)
                P[f, i, j] = win.flat[a]
    v = P.reshape(-1)
    V = np.array(th['V'], float)
    u = V @ v + np.array(th['c'], float)
    e = np.exp(u - u.max())
    pr = e / e.sum()
    L = -np.log(pr[cfg['target']])
    return dict(Xp=Xp, Z=Z, A=A, P=P, arg=arg, v=v, u=u, p=pr, L=L)


def backward(cfg):
    m = forward(cfg)
    H, W, k, oh, ow, ph, pw = shapes(cfg)
    s, p, q = cfg['stride'], cfg['pad'], cfg['q']
    F = len(cfg['filters'])
    V = np.array(cfg['V'], float)
    t = np.zeros(len(cfg['c']))
    t[cfg['target']] = 1
    du = m['p'] - t
    dV = np.outer(du, m['v'])
    dc = du.copy()
    dv = V.T @ du
    dP = dv.reshape(F, ph, pw)
    dA = np.zeros((F, oh, ow))
    for f in range(F):
        for i in range(ph):
            for j in range(pw):
                a, b = m['arg'][f, i, j]
                dA[f, a, b] += dP[f, i, j]
    dZ = dA * (m['Z'] > 0)
    dW = np.zeros((F, k, k))
    db = np.zeros(F)
    for f in range(F):
        for a in range(k):
            for b in range(k):
                for i in range(oh):
                    for j in range(ow):
                        dW[f, a, b] += dZ[f, i, j] * m['Xp'][i * s + a, j * s + b]
        db[f] = dZ[f].sum()
    return m, dict(du=du, dV=dV, dc=dc, dv=dv, dP=dP, dA=dA, dZ=dZ, dW=dW, db=db)


def apply(cfg, g, eta):
    c2 = json.loads(json.dumps(cfg))
    for f in range(len(cfg['filters'])):
        c2['filters'][f]['w'] = (np.array(cfg['filters'][f]['w']) - eta * g['dW'][f]).tolist()
        c2['filters'][f]['b'] = cfg['filters'][f]['b'] - eta * g['db'][f]
    c2['V'] = (np.array(cfg['V']) - eta * g['dV']).tolist()
    c2['c'] = (np.array(cfg['c']) - eta * g['dc']).tolist()
    return c2


def numgrad(cfg, eps=1e-6):
    out = {}
    def loss(c):
        return forward(c)['L']
    def bump(path, d):
        c = json.loads(json.dumps(cfg))
        f, a, b = path
        if a == 'b':
            c['filters'][f]['b'] += d
        else:
            c['filters'][f]['w'][a][b] += d
        return c
    F, k = len(cfg['filters']), len(cfg['filters'][0]['w'])
    for f in range(F):
        for a in range(k):
            for b in range(k):
                out[f'w{f}_{a}_{b}'] = (loss(bump((f, a, b), eps)) - loss(bump((f, a, b), -eps))) / (2 * eps)
        out[f'b{f}'] = (loss(bump((f, 'b', 0), eps)) - loss(bump((f, 'b', 0), -eps))) / (2 * eps)
    for c_ in range(len(cfg['c'])):
        for n in range(len(cfg['V'][0])):
            cp, cm = json.loads(json.dumps(cfg)), json.loads(json.dumps(cfg))
            cp['V'][c_][n] += eps; cm['V'][c_][n] -= eps
            out[f'V{c_}_{n}'] = (loss(cp) - loss(cm)) / (2 * eps)
        cp, cm = json.loads(json.dumps(cfg)), json.loads(json.dumps(cfg))
        cp['c'][c_] += eps; cm['c'][c_] -= eps
        out[f'c{c_}'] = (loss(cp) - loss(cm)) / (2 * eps)
    return out


def pack(cfg):
    m, g = backward(cfg)
    c2 = apply(cfg, g, cfg['eta'])
    m2 = forward(c2)
    ng = numgrad(cfg)
    F, k = len(cfg['filters']), len(cfg['filters'][0]['w'])
    an = {}
    for f in range(F):
        for a in range(k):
            for b in range(k):
                an[f'w{f}_{a}_{b}'] = g['dW'][f, a, b]
        an[f'b{f}'] = g['db'][f]
    for c_ in range(len(cfg['c'])):
        for n in range(len(cfg['V'][0])):
            an[f'V{c_}_{n}'] = g['dV'][c_, n]
        an[f'c{c_}'] = g['dc'][c_]
    cur, hist = cfg, [m['L']]
    for _ in range(200):
        mm, gg = backward(cur)
        cur = apply(cur, gg, cfg['eta'])
        hist.append(forward(cur)['L'])
    J = lambda d: {k_: (v.tolist() if hasattr(v, 'tolist') else v) for k_, v in d.items()}
    return dict(cfg=cfg, fwd=J(m), bwd=J(g), after=J(m2), analytic=an, numeric=ng,
                maxdiff=max(abs(an[x] - ng[x]) for x in an), hist=hist)


if __name__ == '__main__':
    cfgs = {}
    cfgs['default'] = default_cfg()
    # a different custom example: 7x5 image, 2 filters 2x2 (vertical / horizontal edge), stride 1, pad 1, 3 classes
    rng = np.random.default_rng(7)
    img2 = np.round(rng.random((7, 5)), 2).tolist()
    C = dict(img=img2, filters=[dict(w=[[1, -1], [1, -1]], b=0.05), dict(w=[[1, 1], [-1, -1]], b=-0.02)],
             stride=1, pad=1, q=2, V=None, c=[0.0, 0.1, -0.1], target=2, eta=.2)
    _, _, _, oh, ow, ph, pw = shapes(C)
    n = 2 * ph * pw
    C['V'] = np.round(rng.normal(0, .3, (3, n)), 2).tolist()
    cfgs['custom'] = C
    # stride 2 example
    D = default_cfg(); D['stride'] = 2; D['pad'] = 1
    _, _, _, oh, ow, ph, pw = shapes(D)
    D['V'] = np.round(rng.normal(0, .3, (2, ph * pw)), 2).tolist()
    cfgs['stride2'] = D
    out = {k: pack(v) for k, v in cfgs.items()}
    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else 'cnn_ref.json', 'w'))
    for k, v in out.items():
        print(k, 'L=%.6f' % v['fwd']['L'], "L'=%.6f" % v['after']['L'], 'maxdiff=%.2e' % v['maxdiff'], 'hist200=%.6f' % v['hist'][-1])
