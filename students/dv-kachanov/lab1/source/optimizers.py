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


class SGD(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        weight_decay=0.0
    ):
        super().__init__(params)

        self.lr = lr
        self.weight_decay = weight_decay

    def step(self):
        for p in self.params:
            grad = p.grad.copy()

            if self.weight_decay != 0.0:
                regularization = self.weight_decay * p.data
                regularization[0] = 0.0
                grad += regularization

            p.data -= (
                self.lr
                * grad
            )


class SGDMomentum(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0
    ):
        super().__init__(params)

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

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

            if self.weight_decay != 0.0:
                regularization = self.weight_decay * p.data
                regularization[0] = 0.0
                grad += regularization

            self.velocity[i] = (
                self.momentum
                * self.velocity[i]
                +
                (1.0 - self.momentum)
                * grad
            )

            p.data -= (
                self.lr
                * self.velocity[i]
            )
            
