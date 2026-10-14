import numpy as np

class Tensor:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=float)
        self.grad = np.zeros_like(self.data, dtype=float)
        self._prev = ()
        self._op = lambda: None

    def zero_grad(self):
        self.grad.fill(0.0)

    def backward(self):
        topo = []
        visited = set()

        def build(node):
            if id(node) in visited:
                return

            visited.add(id(node))

            for parent in node._prev:
                build(parent)

            topo.append(node)

        build(self)

        for node in topo:
            node.grad = np.zeros_like(
                node.data,
                dtype=float
            )

        self.grad = np.ones_like(
            self.data,
            dtype=float
        )

        for node in reversed(topo):
            node._op()

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        result = Tensor(
            self.data + other.data
        )

        result._prev = (
            self,
            other
        )

        def _backward():
            self.grad += _unbroadcast(
                result.grad,
                self.data.shape
            )

            other.grad += _unbroadcast(
                result.grad,
                other.data.shape
            )

        result._op = _backward

        return result

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        result = Tensor(
            -self.data
        )

        result._prev = (self,)

        def _backward():
            self.grad -= result.grad

        result._op = _backward

        return result

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        result = Tensor(
            self.data * other.data
        )

        result._prev = (
            self,
            other
        )

        def _backward():
            self_grad = (
                result.grad
                * other.data
            )

            other_grad = (
                result.grad
                * self.data
            )

            self.grad += _unbroadcast(
                self_grad,
                self.data.shape
            )

            other.grad += _unbroadcast(
                other_grad,
                other.data.shape
            )

        result._op = _backward

        return result

    def __matmul__(self, other):

        if not isinstance(other, Tensor):
            other = Tensor(other)

        if not (
            self.data.ndim == 2
            and other.data.ndim == 1
        ):
            raise NotImplementedError(
                "Поддерживается только matrix @ vector: "
                f"получено {self.data.shape} @ "
                f"{other.data.shape}"
            )

        result = Tensor(
            self.data @ other.data
        )

        result._prev = (
            self,
            other
        )

        def _backward():
            self.grad += np.outer(
                result.grad,
                other.data
            )

            other.grad += (
                self.data.T
                @ result.grad
            )

        result._op = _backward

        return result

    def exp(self):
        result = Tensor(
            np.exp(self.data)
        )

        result._prev = (self,)

        def _backward():
            self.grad += (
                result.data
                * result.grad
            )

        result._op = _backward

        return result

    def log(self):
        result = Tensor(
            np.log(self.data)
        )

        result._prev = (self,)

        def _backward():
            self.grad += (
                result.grad
                / self.data
            )

        result._op = _backward

        return result

    def mean(self):
        result = Tensor(
            np.mean(self.data)
        )

        result._prev = (self,)

        def _backward():
            self.grad += (
                np.ones_like(
                    self.data,
                    dtype=float
                )
                * result.grad
                / self.data.size
            )

        result._op = _backward

        return result

def _unbroadcast(grad, shape):
    grad = np.asarray(grad, dtype=float)

    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad
