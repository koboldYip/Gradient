from abc import ABC, abstractmethod

from sympy import symbols


class Gradient(ABC):
    w1 = symbols('w1')
    w2 = symbols('w2')
    gradient_w1 = None
    gradient_w2 = None
    rate = None
    epsilon = None
    max_iter = None
    lambda_ = None
    delta = None
    Delta = None
    alpha = None

    @abstractmethod
    def process(self):
        pass
