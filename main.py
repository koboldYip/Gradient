import matplotlib.pyplot as plt
from sympy import *

from AdaDelta import AdaDelta
from Adam import Adam
from Classic import Classic
from Momentum import Momentum
from NAG import NAG
from RMSProp import RMSProp

rate = 0.01  # Скорость спуска
epsilon = 0.000001  # Точность
max_iter = 10000  # Максимальное количество итераций
lambda_ = 0.1
delta = 0.01
Delta = 0.01
alpha = 0.999

# Исходная функция
w1 = symbols('w1')
w2 = symbols('w2')
f = 8 * w1 ** 2 + 4.4 * w2 ** 2 - 3 * w1 - 7.2 * w2

# Нахождение частных производных
Gradient_W1 = diff(f, w1)
Gradient_W2 = diff(f, w2)

classic = Classic(Gradient_W1, Gradient_W2, rate, epsilon, max_iter)
momentum = Momentum(Gradient_W1, Gradient_W2, rate, epsilon, max_iter, lambda_)
nag = NAG(Gradient_W1, Gradient_W2, rate, epsilon, max_iter, lambda_)
rms = RMSProp(Gradient_W1, Gradient_W2, epsilon, max_iter, lambda_)
ada = AdaDelta(Gradient_W1, Gradient_W2, alpha, epsilon, max_iter, delta, Delta)
adam = Adam(Gradient_W1, Gradient_W2, alpha, rate, lambda_, epsilon, max_iter)

# Вызываем из классов метода для нахождения минимума функции классическим градиентным методом

Grad_Iteration_Classic, Grad_W1_Classic, Grad_W2_Classic = classic.process()
Grad_Iteration_Momentum, Grad_W1_Momentum, Grad_W2_Momentum = momentum.process()
Grad_Iteration_NAG, Grad_W1_NAG, Grad_W2_NAG = nag.process()
Grad_Iteration_RMSProp, Grad_W1_RMSProp, Grad_W2_RMSProp = rms.process()
Grad_Iteration_AdaDelta, Grad_W1_AdaDelta, Grad_W2_AdaDelta = ada.process()
Grad_Iteration_Adam, Grad_W1_Adam, Grad_W2_Adam = adam.process()

iteration = [Grad_Iteration_Classic, Grad_Iteration_Momentum, Grad_Iteration_NAG, Grad_Iteration_RMSProp,
             Grad_Iteration_AdaDelta, Grad_Iteration_Adam]
w1 = [Grad_W1_Classic, Grad_W1_Momentum, Grad_W1_NAG, Grad_W1_RMSProp, Grad_W1_AdaDelta, Grad_W1_Adam]
w2 = [Grad_W2_Classic, Grad_W2_Momentum, Grad_W2_NAG, Grad_W2_RMSProp, Grad_W2_AdaDelta, Grad_W2_Adam]

# iteration = [Grad_Iteration_Classic, Grad_Iteration_Momentum, Grad_Iteration_NAG]
# w1 = [Grad_W1_Classic, Grad_W1_Momentum, Grad_W1_NAG]
# w2 = [Grad_W2_Classic, Grad_W2_Momentum, Grad_W2_NAG]

# iteration = [Grad_Iteration_Classic, Grad_Iteration_Momentum, Grad_Iteration_NAG, Grad_Iteration_Adam]
# w1 = [Grad_W1_Classic, Grad_W1_Momentum, Grad_W1_NAG, Grad_W1_Adam]
# w2 = [Grad_W2_Classic, Grad_W2_Momentum, Grad_W2_NAG, Grad_W2_Adam]

plt.figure(0)
plt.boxplot(iteration)
plt.ylabel("Iteration")

plt.figure(1)
plt.boxplot(w1)
plt.ylabel("w1")

plt.figure(2)
plt.boxplot(w1)
plt.ylabel("w2")

plt.show()
