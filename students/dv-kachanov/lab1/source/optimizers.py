from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    def __init__(self, params):
        self.params = list(params)

    @abstractmethod
    def step(self):
        pass

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()


class SGDMomentum(Optimizer):
    """Стохастический градиент с инерцией (лекция 1).

    v := gamma * v + (1 - gamma) * grad ~L(w, x_i)
    w := w - h * v

    Регуляризация здесь НЕ добавляется: штраф tau/2 * ||w||^2
    входит в граф функции потерь ~L, поэтому grad уже содержит
    слагаемое tau * w по всем компонентам, включая w_0.
    """

    def __init__(
        self,
        params,
        h=0.01,
        gamma=0.9
    ):
        super().__init__(params)

        self.h = h
        self.gamma = gamma

        self.velocity = [
            np.zeros_like(
                p.data,
                dtype=float
            )
            for p in self.params
        ]

    def step(self):
        for i, p in enumerate(self.params):
            grad = p.grad.copy()

            self.velocity[i] = (
                self.gamma
                * self.velocity[i]
                +
                (1.0 - self.gamma)
                * grad
            )

            p.data -= (
                self.h
                * self.velocity[i]
            )
