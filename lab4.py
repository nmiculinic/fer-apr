import numpy as np
from functools import reduce
from copy import deepcopy


class Chromosone():

    def mutate(self, step=None):
        raise NotImplemented

    def eval(self):
        raise NotImplemented

    def crossover(self, other):
        raise NotImplemented


class BinaryChromosone(Chromosone):

    def __init__(self, params):
        self.l = params['lower']
        self.u = params['upper']
        prec = params['precision']
        degree = params['degree']
        self.n = np.int(np.ceil(np.log2(np.floor(1 + (self.u - self.l) * 10**prec))))
        self.data = np.random.randint(2, size=[degree, self.n])

        crossover_fns = {
            'uniform': self._uniform_crossover
        }

        self.pc = params['pc']
        self.pm = params['pm']
        self.crossover_fn = crossover_fns[params['crossover_fn']]
        self.degree = degree

    def eval(self):
        return (self.u - self.l) / (2**self.n - 1) * reduce(lambda x, y: 2 * x + y, self.data.T) + self.l

    def _uniform_crossover(self, other):
        sol = deepcopy(self)
        sol.data = np.where(np.random.uniform(size=self.data.shape), self.data, other.data)
        return sol

    def crossover(self, other):
        return self.crossover_fn(other)

    def mutate(self, step=None, maxStep=None):
        sol = deepcopy(self)
        sol.data = np.where(np.random.uniform(size=(self.degree, self.n)) < self.pm, 1 - self.data, self.data)
        return sol


class RealChromosone(Chromosone):

    def __init__(self, params):
        self.l = params['lower']
        self.u = params['upper']
        degree = params['degree']
        self.data = np.random.uniform(self.l, self.u, size=degree)

        crossover_fns = {
            'uniform': self._uniform_crossover
        }

        self.b = params['b']
        self.crossover_fn = crossover_fns[params['crossover_fn']]
        self.degree = degree

    def eval(self):
        return self.data

    def _uniform_crossover(self, other):
        sol = deepcopy(self)
        alpha = np.random.uniform(size=self.data.shape)
        sol.data = alpha * self.data + (1 - alpha) * other.data
        return sol

    def _full_cromosson():
        pass

    def crossover(self, other):
        return self.crossover_fn(other)

    def mutate(self, step=None, maxStep=None):
        sol = deepcopy(self)
        r = 1 - np.random.uniform(size=1)**(1 - step / maxStep)**self.b
        noise = (self.u - self.l) * np.random.uniform(-r, +r, size=self.degree)
        sol.data = np.clip(sol.data + noise, self.l, self.u)
        return sol



params = {
    'n': 200,
    'chromosome': {
        'class': BinaryChromosone,
        'params': {
            'lower': -50,
            'upper': 150,
            'degree': 5,
            'precision': 5,  # Broj decimala
            'crossover_fn': 'uniform',
            'pc': 0.5,
            'pm': 0.3
        }
    }
}

params = {
    'n': 200,
    'chromosome': {
        'class': RealChromosone,
        'params': {
            'lower': -50,
            'upper': 150,
            'degree': 5,
            'crossover_fn': 'uniform',
            'b': 5
        }
    }
}

class GA(object):
    def __init__(self, params):
        self.n = int(params['n'])
        ch_data = params['chromosome']
        self.population = [
            ch_data['class'](ch_data['params']) for _ in range(self.n)]

    def _props(self):
        probs = np.copy(self.vals)
        probs = (np.max(probs) - probs)/np.max(probs)
        probs /= np.sum(probs)
        return probs

    def _generation_opt(self, f, step, maxSteps):

        elems = np.random.choice(self.population, size=2 * (self.n - 2), replace=True, p=self._props())

        self.population[-1] = self.population[np.argmin(self.vals)]
        self.population[-2] = self.population[-1].mutate(step, maxSteps)

        for i in range(self.n - 2):
            a = elems[2*i]
            b = elems[2*i + 1]
            self.population[i] = a.crossover(b).mutate(step, maxSteps)
        self.vals = np.array([f(x.eval()) for x in self.population]).ravel()

    def _tournament_opt(self, f, step, maxSteps):
        for _ in range(self.n):
            idx = np.random.choice(np.arange(self.n), size=3, replace=False, p=self._props())
            order = np.argsort(self.vals[idx])
            idx = idx[order]

            self.population[idx[1]] = self.population[idx[0]].crossover(self.population[1]).mutate(step, maxSteps)
            self.vals[idx[1]] = f(self.population[idx[1]].eval())

    def optimize(self, f, steps=200, full=False):
        self.vals = np.array([f(x.eval()) for x in self.population]).ravel()
        for i in range(steps):
            self.pr(f, i)
            self._generation_opt(f, i, steps)
            # self._tournament_opt(f, i, steps)
        self.pr(f, steps, full)

    def pr(self, f, step=None, full=False):
        if step is not None:
            print("Step: ", step)
        if full:
            for x in self.population:
                print(x.eval(), f(x.eval()))
        else:
            x = np.argmin(self.vals)
            print(self.population[x].eval(), self.vals[x])


g = GA(params)
g.optimize(lambda x: np.sum(x**2 + 4), full=False)
