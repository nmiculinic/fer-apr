from functools import lru_cache
import numpy as np
import math
gr = (math.sqrt(5) + 1) / 2


def gss(f, a, b, eps=1e-6, debug=False):
    f = lru_cache(10)(f)
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    while abs(a - b) > eps:
        if debug:
            print(a, b, c, d)
            print(f(a), f(b), f(c), f(d))

        if f(c) < f(d):
            b = d
        else:
            a = c

        c = b - (b - a) / gr
        d = a + (b - a) / gr

    return (b + a) / 2


def unimodalni(f, h, x):
    f = lru_cache(10)(f)
    l, r = x - h, x + h
    step = 1

    if f(l) > f(x) < f(r):
        return (l, r)

    if f(l) < f(x) < f(r):
        while f(l) < f(x):
            l -= step * h
            step *= 2
        return (l, r)
    else:  # f(l) > f(x) > f(r):
        while f(r) < f(x):
            r += step * h
            step *= 2
        return (l, r)

# print(gss(lambda x: (x - 2)**2, 1, 5))


def coordinate_descent(f, x_0, eps=1e-6):
    x = np.array(x_0)
    n = x.shape[0]
    x_prev = x + 2 * eps
    while np.linalg.norm(x_prev - x) > eps:
        x_prev = np.copy(x)
        for i in range(n):
            def fi(xt):
                xp = np.copy(x)
                xp[i] = xt
                return f(xp)
            l, r = unimodalni(fi, 0.1, x[i])
            x[i] = gss(fi, l, r, eps)
    return x

def hook_jeeves(f, x_0, eps=1e-6):
    def explore(x, dx):
        x = np.copy(x)
        for i, val in enumerate(x):
            P = f(x)
            x[i] += dx[i]
            if f(x) > P:
                x[i] -= 2 * dx[i]
                if f(x) > P:
                    x[i] += dx[i]
        return x

    dx = 0.5 * np.ones_like(x_0)
    esp = eps * np.ones_like(x_0)

    xp = np.copy(x_0)
    xb = np.copy(x_0)

    while True:
        xn = explore(xp, dx)

        if f(xn) < f(xb):
            xp = 2 * xn - xb
            xb = xn
        else:
            dx /= 2
            if np.any(esp > dx):
                return xb
            xp = xb


def nelde_mead(f, x_0, alpha=1, beta=0.5, gamma=2, sigma=0.5, eps=1e-6, maxIter=1000, shift=1, trace=False):
    X = [x_0]
    n = x_0.shape[0]
    for i in range(n):
        xp = np.copy(x_0)
        xp[i] += shift
        X.append(xp)
    X = np.array(X)

    iter = 0
    while np.max(np.linalg.norm(X - np.mean(X, axis=0), axis=0)) > eps and iter < maxIter:
        iter += 1
        vals = list(map(f, X))
        if trace:
            print("X\n", X)
            print("f(X) = \n", vals)
        l = np.argmin(vals)
        h = np.argmax(vals)

        rr = list(range(n + 1))
        rr.remove

        Xc = np.mean(np.concatenate([X[:h], X[h + 1:]]), axis=0)

        Xr = Xc + alpha * (Xc - X[h])
        if f(Xr) < f(X[l]):
            Xe = Xc + gamma * (Xc - X[h])
            X[h] = Xe if f(Xe) < f(X[l]) else Xr
        else:
            vals.pop(h)
            if np.all(f(Xr) > np.array(vals)):
                if f(Xr) < f(X[h]):
                    X[h] = Xr

                # Kontrakcija
                Xk = Xc - beta * (Xc - X[h])
                if f(Xk) < f(X[h]):
                    X[h] = Xk
                else:
                    for i in range(n):
                        X[i] = sigma * X[i] + (1 - sigma) * X[l]
            else:
                X[h] = Xr
    return X[0]


class Call_counter:
    def __init__(self, f):
        self.f = f
        self.called = set()

    def __call__(self, x):
        self.called.add(tuple(x))
        return self.f(x)

    def count(self):
        return len(self.called)



def f1(x):
    x1 = x[0]
    x2 = x[1]
    return 100 * (x2 - x1**2)**2 + (1 - x1)**2

f1_0 = np.array([-1.9, 2])

# hook_jeeves(f1, f1_0)
# nelde_mead(f1, f1_0)

def f2(x):
    x1, x2 = x
    return (x1 - 4)**2 + 4 * (x2 - 2)**2
f2_0 = np.array([0.1, 0.3])


def f3(x):
    return np.sum(np.square(np.arange(1, x.shape[0] + 1) - x))

def f4(x):
    x1, x2 = x
    return abs((x1 - x2)*(x1 + x2)) + math.sqrt(x1**2 + x2**2)
f4_0 = np.array([5.1, 1.1])

def f6(x):
    x2 = np.sum(np.square(x))
    return 0.5 + (math.sin(math.sqrt(x2)) - 0.5)/(1 + 0.001*x2)**2

def f(x):
    return (x - 2)**2

# print(unimodalni(f, 0.1, 0))
# print(unimodalni(f, 0.1, 2.01))
# print(unimodalni(f, 0.1, 5))
