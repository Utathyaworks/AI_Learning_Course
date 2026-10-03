import math

class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError
