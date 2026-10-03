import math

class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @staticmethod
    def _wrap(o):
        return o if isinstance(o, Value) else Value(o)

    def __add__(self, other):
        other = self._wrap(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _b():
            self.grad += out.grad; other.grad += out.grad
        out._backward = _b
        return out

    def __mul__(self, other):
        other = self._wrap(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _b():
            self.grad += other.data * out.grad; other.grad += self.data * out.grad
        out._backward = _b
        return out

    def __pow__(self, k):
        out = Value(self.data ** k, (self,), "**")
        def _b():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _b
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def _b():
            self.grad += (1 - t * t) * out.grad
        out._backward = _b
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")
        def _b():
            self.grad += e * out.grad
        out._backward = _b
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")
        def _b():
            self.grad += out.grad / self.data
        out._backward = _b
        return out

    def __neg__(self): return self * -1
    def __sub__(self, o): return self + (-self._wrap(o))
    def __rsub__(self, o): return self._wrap(o) - self
    def __radd__(self, o): return self + o
    def __rmul__(self, o): return self * o
    def __truediv__(self, o): return self * self._wrap(o) ** -1
    def __rtruediv__(self, o): return self._wrap(o) * self ** -1

    def backward(self):
        topo, seen = [], set()
        def build(v):
            if v not in seen:
                seen.add(v)
                for c in v._prev:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
