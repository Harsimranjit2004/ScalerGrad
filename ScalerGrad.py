import math
import random
from typing import Union, Tuple, Callable, List

class Scalar:
    """A scalar or tensor value with automatic differentiation support for computational graphs."""

    def __init__(self, data: Union[float, List], _children=(), _op: str = '', label: str = ''):
        if isinstance(data, (int, float)):
            self.data = float(data)
            self.shape = ()
            self.grad = 0.0
        elif isinstance(data, list):
            self.data = data 
            if not data:
                self.shape = ()
                self.grad = []
            elif isinstance(data[0], list):
                self.shape = (len(data), len(data[0]))
                self.grad = [[0.0 for _ in range(len(data[0]))] for _ in range(len(data))]
            else:
                self.shape = (len(data),)
                self.grad = [0.0 for _ in range(len(data))]
        else:
            raise ValueError("Data must be a scalar or list")
        
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        
        if self.shape == () and other.shape != ():
            if len(other.shape) == 1:
                out_data = [self.data + other.data[i] for i in range(other.shape[0])]
            else:
                out_data = [[self.data + other.data[i][j] for j in range(other.shape[1])] for i in range(other.shape[0])]
            out = Scalar(out_data, (self, other), '+')
        elif self.shape != () and other.shape == ():
            if len(self.shape) == 1:
                out_data = [self.data[i] + other.data for i in range(self.shape[0])]
            else:
                out_data = [[self.data[i][j] + other.data for j in range(self.shape[1])] for i in range(self.shape[0])]
            out = Scalar(out_data, (self, other), '+')
        elif self.shape == other.shape:
            if self.shape == ():
                out_data = self.data + other.data
            elif len(self.shape) == 1:
                out_data = [self.data[i] + other.data[i] for i in range(self.shape[0])]
            else:
                out_data = [[self.data[i][j] + other.data[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])]
            out = Scalar(out_data, (self, other), '+')
        else:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")

        def _backward():
            if self.shape == ():
                if isinstance(out.grad, (int, float)):
                    self.grad += out.grad
                else:
                    grad_sum = 0.0
                    if isinstance(out.grad, list) and isinstance(out.grad[0], list):
                        for row in out.grad:
                            grad_sum += sum(row)
                    else:
                        grad_sum = sum(out.grad)
                    self.grad += grad_sum
            elif len(self.shape) == 1:
                for i in range(self.shape[0]):
                    self.grad[i] += out.grad[i] if isinstance(out.grad, list) else out.grad
            else:
                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        self.grad[i][j] += out.grad[i][j] if isinstance(out.grad, list) else out.grad
                        
            if other.shape == ():
                if isinstance(out.grad, (int, float)):
                    other.grad += out.grad
                else:
                    grad_sum = 0.0
                    if isinstance(out.grad, list) and isinstance(out.grad[0], list):
                        for row in out.grad:
                            grad_sum += sum(row)
                    else:
                        grad_sum = sum(out.grad)
                    other.grad += grad_sum
            elif len(other.shape) == 1:
                for i in range(other.shape[0]):
                    other.grad[i] += out.grad[i] if isinstance(out.grad, list) else out.grad
            else:
                for i in range(other.shape[0]):
                    for j in range(other.shape[1]):
                        other.grad[i][j] += out.grad[i][j] if isinstance(out.grad, list) else out.grad
        
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        
        if self.shape == () and other.shape != ():
            if len(other.shape) == 1:
                out_data = [self.data * other.data[i] for i in range(other.shape[0])]
            else:
                out_data = [[self.data * other.data[i][j] for j in range(other.shape[1])] for i in range(other.shape[0])]
            out = Scalar(out_data, (self, other), '*')
        elif self.shape != () and other.shape == ():
            if len(self.shape) == 1:
                out_data = [self.data[i] * other.data for i in range(self.shape[0])]
            else:
                out_data = [[self.data[i][j] * other.data for j in range(self.shape[1])] for i in range(self.shape[0])]
            out = Scalar(out_data, (self, other), '*')
        elif self.shape == other.shape:
            if self.shape == ():
                out_data = self.data * other.data
            elif len(self.shape) == 1:
                out_data = [self.data[i] * other.data[i] for i in range(self.shape[0])]
            else:
                out_data = [[self.data[i][j] * other.data[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])]
            out = Scalar(out_data, (self, other), '*')
        else:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")

        def _backward():
            if self.shape == ():
                if other.shape == ():
                    self.grad += other.data * out.grad
                else:
                    grad_sum = 0.0
                    if len(other.shape) == 1:
                        for i in range(other.shape[0]):
                            grad_sum += other.data[i] * out.grad[i]
                    else:
                        for i in range(other.shape[0]):
                            for j in range(other.shape[1]):
                                grad_sum += other.data[i][j] * out.grad[i][j]
                    self.grad += grad_sum
            elif len(self.shape) == 1:
                for i in range(self.shape[0]):
                    other_val = other.data if other.shape == () else other.data[i]
                    grad_val = out.grad[i] if isinstance(out.grad, list) else out.grad
                    self.grad[i] += other_val * grad_val
            else:
                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        other_val = other.data if other.shape == () else other.data[i][j]
                        grad_val = out.grad[i][j] if isinstance(out.grad, list) else out.grad
                        self.grad[i][j] += other_val * grad_val
                        
            if other.shape == ():
                if self.shape == ():
                    other.grad += self.data * out.grad
                else:
                    grad_sum = 0.0
                    if len(self.shape) == 1:
                        for i in range(self.shape[0]):
                            grad_sum += self.data[i] * out.grad[i]
                    else:
                        for i in range(self.shape[0]):
                            for j in range(self.shape[1]):
                                grad_sum += self.data[i][j] * out.grad[i][j]
                    other.grad += grad_sum
            elif len(other.shape) == 1:
                for i in range(other.shape[0]):
                    self_val = self.data if self.shape == () else self.data[i]
                    grad_val = out.grad[i] if isinstance(out.grad, list) else out.grad
                    other.grad[i] += self_val * grad_val
            else:
                for i in range(other.shape[0]):
                    for j in range(other.shape[1]):
                        self_val = self.data if self.shape == () else self.data[i][j]
                        grad_val = out.grad[i][j] if isinstance(out.grad, list) else out.grad
                        other.grad[i][j] += self_val * grad_val
        
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return self + (-1 * other)

    def __rsub__(self, other):
        return other + (-1 * self)

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)

    def __neg__(self):
        return self * -1

    def dot(self, other):
        """Dot product for 1D vectors."""
        other = other if isinstance(other, Scalar) else Scalar(other)
        if len(self.shape) != 1 or len(other.shape) != 1:
            raise ValueError("Dot product requires 1D vectors")
        if self.shape[0] != other.shape[0]:
            raise ValueError(f"Vector dimensions must match: {self.shape[0]} vs {other.shape[0]}")
        
        out_data = sum(self.data[i] * other.data[i] for i in range(self.shape[0]))
        out = Scalar(out_data, (self, other), 'dot')

        def _backward():
            for i in range(self.shape[0]):
                self.grad[i] += other.data[i] * out.grad
                other.grad[i] += self.data[i] * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only supporting int/float powers"
        
        if self.shape == ():
            out_data = self.data ** other
        elif len(self.shape) == 1:
            out_data = [self.data[i] ** other for i in range(self.shape[0])]
        else:
            out_data = [[self.data[i][j] ** other for j in range(self.shape[1])] for i in range(self.shape[0])]
        
        out = Scalar(out_data, (self,), f'**{other}')

        def _backward():
            if self.shape == ():
                self.grad += (other * self.data ** (other - 1)) * out.grad
            elif len(self.shape) == 1:
                for i in range(self.shape[0]):
                    grad_val = out.grad[i] if isinstance(out.grad, list) else out.grad
                    self.grad[i] += (other * self.data[i] ** (other - 1)) * grad_val
            else:
                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        grad_val = out.grad[i][j] if isinstance(out.grad, list) else out.grad
                        self.grad[i][j] += (other * self.data[i][j] ** (other - 1)) * grad_val
        
        out._backward = _backward
        return out

    def relu(self):
        if self.shape == ():
            out_data = 0.0 if self.data < 0 else self.data
        elif len(self.shape) == 1:
            out_data = [0.0 if self.data[i] < 0 else self.data[i] for i in range(self.shape[0])]
        else:
            out_data = [[0.0 if self.data[i][j] < 0 else self.data[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])]
        
        out = Scalar(out_data, (self,), 'ReLU')

        def _backward():
            if self.shape == ():
                self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
            elif len(self.shape) == 1:
                for i in range(self.shape[0]):
                    grad_val = out.grad[i] if isinstance(out.grad, list) else out.grad
                    self.grad[i] += (1.0 if out.data[i] > 0 else 0.0) * grad_val
            else:
                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        grad_val = out.grad[i][j] if isinstance(out.grad, list) else out.grad
                        self.grad[i][j] += (1.0 if out.data[i][j] > 0 else 0.0) * grad_val
        
        out._backward = _backward
        return out

    def tanh(self):
        if self.shape == ():
            t = (math.exp(2*self.data) - 1) / (math.exp(2*self.data) + 1)
            out_data = t
        elif len(self.shape) == 1:
            out_data = [(math.exp(2*self.data[i]) - 1) / (math.exp(2*self.data[i]) + 1) for i in range(self.shape[0])]
        else:
            out_data = [[(math.exp(2*self.data[i][j]) - 1) / (math.exp(2*self.data[i][j]) + 1) for j in range(self.shape[1])] for i in range(self.shape[0])]
        
        out = Scalar(out_data, (self,), 'tanh')

        def _backward():
            if self.shape == ():
                self.grad += (1 - out.data**2) * out.grad
            elif len(self.shape) == 1:
                for i in range(self.shape[0]):
                    grad_val = out.grad[i] if isinstance(out.grad, list) else out.grad
                    self.grad[i] += (1 - out.data[i]**2) * grad_val
            else:
                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        grad_val = out.grad[i][j] if isinstance(out.grad, list) else out.grad
                        self.grad[i][j] += (1 - out.data[i][j]**2) * grad_val
        
        out._backward = _backward
        return out

    def sigmoid(self):
        if self.shape == ():
            s = 1 / (1 + math.exp(-self.data))
            out_data = s
        elif len(self.shape) == 1:
            out_data = [1 / (1 + math.exp(-self.data[i])) for i in range(self.shape[0])]
        else:
            out_data = [[1 / (1 + math.exp(-self.data[i][j])) for j in range(self.shape[1])] for i in range(self.shape[0])]
        
        out = Scalar(out_data, (self,), 'sigmoid')

        def _backward():
            if self.shape == ():
                self.grad += out.data * (1 - out.data) * out.grad
            elif len(self.shape) == 1:
                for i in range(self.shape[0]):
                    grad_val = out.grad[i] if isinstance(out.grad, list) else out.grad
                    self.grad[i] += out.data[i] * (1 - out.data[i]) * grad_val
            else:
                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        grad_val = out.grad[i][j] if isinstance(out.grad, list) else out.grad
                        self.grad[i][j] += out.data[i][j] * (1 - out.data[i][j]) * grad_val
        
        out._backward = _backward
        return out

    def exp(self):
        if self.shape == ():
            out_data = math.exp(self.data)
        elif len(self.shape) == 1:
            out_data = [math.exp(self.data[i]) for i in range(self.shape[0])]
        else:
            out_data = [[math.exp(self.data[i][j]) for j in range(self.shape[1])] for i in range(self.shape[0])]
        
        out = Scalar(out_data, (self,), 'exp')

        def _backward():
            if self.shape == ():
                self.grad += out.data * out.grad
            elif len(self.shape) == 1:
                for i in range(self.shape[0]):
                    grad_val = out.grad[i] if isinstance(out.grad, list) else out.grad
                    self.grad[i] += out.data[i] * grad_val
            else:
                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        grad_val = out.grad[i][j] if isinstance(out.grad, list) else out.grad
                        self.grad[i][j] += out.data[i][j] * grad_val
        
        out._backward = _backward
        return out

    def log(self):
        if self.shape == ():
            assert self.data > 0, "Logarithm undefined for non-positive values"
            out_data = math.log(self.data)
        elif len(self.shape) == 1:
            for x in self.data:
                assert x > 0, "Logarithm undefined for non-positive values"
            out_data = [math.log(self.data[i]) for i in range(self.shape[0])]
        else:
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    assert self.data[i][j] > 0, "Logarithm undefined for non-positive values"
            out_data = [[math.log(self.data[i][j]) for j in range(self.shape[1])] for i in range(self.shape[0])]
        
        out = Scalar(out_data, (self,), 'log')

        def _backward():
            if self.shape == ():
                self.grad += (1 / self.data) * out.grad
            elif len(self.shape) == 1:
                for i in range(self.shape[0]):
                    grad_val = out.grad[i] if isinstance(out.grad, list) else out.grad
                    self.grad[i] += (1 / self.data[i]) * grad_val
            else:
                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        grad_val = out.grad[i][j] if isinstance(out.grad, list) else out.grad
                        self.grad[i][j] += (1 / self.data[i][j]) * grad_val
        
        out._backward = _backward
        return out

    def sin(self):
        if self.shape == ():
            out_data = math.sin(self.data)
        elif len(self.shape) == 1:
            out_data = [math.sin(self.data[i]) for i in range(self.shape[0])]
        else:
            out_data = [[math.sin(self.data[i][j]) for j in range(self.shape[1])] for i in range(self.shape[0])]
        
        out = Scalar(out_data, (self,), 'sin')

        def _backward():
            if self.shape == ():
                self.grad += math.cos(self.data) * out.grad
            elif len(self.shape) == 1:
                for i in range(self.shape[0]):
                    grad_val = out.grad[i] if isinstance(out.grad, list) else out.grad
                    self.grad[i] += math.cos(self.data[i]) * grad_val
            else:
                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        grad_val = out.grad[i][j] if isinstance(out.grad, list) else out.grad
                        self.grad[i][j] += math.cos(self.data[i][j]) * grad_val
        
        out._backward = _backward
        return out

    def cos(self):
        if self.shape == ():
            out_data = math.cos(self.data)
        elif len(self.shape) == 1:
            out_data = [math.cos(self.data[i]) for i in range(self.shape[0])]
        else:
            out_data = [[math.cos(self.data[i][j]) for j in range(self.shape[1])] for i in range(self.shape[0])]
        
        out = Scalar(out_data, (self,), 'cos')

        def _backward():
            if self.shape == ():
                self.grad += -math.sin(self.data) * out.grad
            elif len(self.shape) == 1:
                for i in range(self.shape[0]):
                    grad_val = out.grad[i] if isinstance(out.grad, list) else out.grad
                    self.grad[i] += -math.sin(self.data[i]) * grad_val
            else:
                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        grad_val = out.grad[i][j] if isinstance(out.grad, list) else out.grad
                        self.grad[i][j] += -math.sin(self.data[i][j]) * grad_val
        
        out._backward = _backward
        return out

    def backward(self, clip_grad: float = None, accumulate: bool = True):
        """Perform backpropagation through the computation graph."""
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)

        if not accumulate:
            for v in topo:
                if v.shape == ():
                    v.grad = 0.0
                elif len(v.shape) == 1:
                    v.grad = [0.0] * v.shape[0]
                else:
                    v.grad = [[0.0] * v.shape[1] for _ in range(v.shape[0])]
        
        if self.shape == ():
            self.grad = 1.0
        elif len(self.shape) == 1:
            self.grad = [1.0] * self.shape[0]
        else:
            self.grad = [[1.0] * self.shape[1] for _ in range(self.shape[0])]

        for v in reversed(topo):
            v._backward()
            
            if clip_grad is not None:
                if v.shape == ():
                    v.grad = max(min(v.grad, clip_grad), -clip_grad)
                elif len(v.shape) == 1:
                    for i in range(v.shape[0]):
                        v.grad[i] = max(min(v.grad[i], clip_grad), -clip_grad)
                else:
                    for i in range(v.shape[0]):
                        for j in range(v.shape[1]):
                            v.grad[i][j] = max(min(v.grad[i][j], clip_grad), -clip_grad)

    def zero_grad(self):
        """Reset gradients to zero."""
        if self.shape == ():
            self.grad = 0.0
        elif len(self.shape) == 1:
            self.grad = [0.0] * self.shape[0]
        else:
            self.grad = [[0.0] * self.shape[1] for _ in range(self.shape[0])]

    def __repr__(self):
        data_str = str(self.data) if self.shape == () else str(self.data)[:50] + "..." if len(str(self.data)) > 50 else str(self.data)
        grad_str = str(self.grad) if self.shape == () else str(self.grad)[:50] + "..." if len(str(self.grad)) > 50 else str(self.grad)
        return f"Scalar(data={data_str}, grad={grad_str}, shape={self.shape})"