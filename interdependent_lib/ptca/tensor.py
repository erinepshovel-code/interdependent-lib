# GPT/Claude generated; context, prompt Erin Spencer
# Date: 2026-04-06

import numpy as np

class Tensor:
    def __init__(self, data):
        self.data = np.array(data)

    def __add__(self, other):
        return Tensor(self.data + other.data)

    def __sub__(self, other):
        return Tensor(self.data - other.data)

    def __mul__(self, other):
        return Tensor(self.data * other.data)

    def __truediv__(self, other):
        return Tensor(self.data / other.data)

    def __repr__(self):
        return f"Tensor({self.data})"