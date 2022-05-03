from math import pi
from math import sin, cos
from itertools import combinations
from ssl import ALERT_DESCRIPTION_INSUFFICIENT_SECURITY
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def sin_nx(n, x):
    return sin(n * x)


def cos_nx(n, x):
    return cos(n * x)


class Function():
    def __init__(self,function:callable):
        self.fx = function
        self.computed_fourier = False


    def _integral(self, func = None, n=None, div = 200):
        area = 0
        x_i = -pi
        if func is None:
            for i in range(div):
                div_area = self.fx(x_i) * (2 * pi / div)
                area += div_area
                x_i += 2 * pi / div
        else:
            for i in range(div):
                div_area = func(n, x_i) * self.fx(x_i) * (2 * pi / div)
                area += div_area
                x_i += 2 * pi / div
        return area
    

    def compute_fourier(self, n = 10):
        self.a = {}
        self.b = {}
        self.a[0] = (1 / pi) * self._integral()
        self.b[0] = None
        for i in tqdm(range(1, n + 1)):
            self.a[i] = 1 / pi * self._integral(cos_nx, i)
            self.b[i] = 1 / pi * self._integral(sin_nx, i)
        self.computed_fourier = True
        

    def fourier(self, x):
        value = 0
        for i, (a_i, b_i) in enumerate(zip(self.a.values(), self.b.values())):
            if i == 0:
                continue
            value += a_i * cos_nx(i, x) + b_i * sin_nx(i, x)
        return value + self.a[0] / 2


    def show(self):
        assert self.computed_fourier is True, "fourier series should be computed before showing graph"
        x = np.linspace(-pi, pi, 200)
        y_fx = [self.fx(x_i) for x_i in x]
        y_fourier = [self.fourier(x_i) for x_i in x]
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set(xlabel='x', ylabel='y')
        ax.grid()
        ax.plot(x, y_fx, label='f(x)', linestyle='dashed')
        ax.plot(x, y_fourier, label='fourier')
        fig.legend()
        plt.show()
