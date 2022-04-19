import random

import numpy as np
from sympy import *

from Gradient import Gradient


class Adam(Gradient):

    def __init__(self, gradient_w1, gradient_w2, alpha, rate, lambda_, epsilon, max_iter):
        self.gradient_w1 = gradient_w1
        self.gradient_w2 = gradient_w2
        self.epsilon = epsilon
        self.rate = rate
        self.max_iter = max_iter
        self.lambda_ = lambda_
        self.alpha = alpha

    # Задание функции для решения градиентным методом
    def process(self):
        array_iteration = np.zeros(100)
        array_w1 = np.zeros(100)
        array_w2 = np.zeros(100)
        for i in range(100):
            w1rand = random.random()
            w2rand = random.random()
            w = np.array([w1rand, w2rand])
            print("Исходная данные для метода Adam:")
            print("w1 = ", w1rand, "w2 = ", w2rand)
            delta = 100
            iteration = 0
            g1rand = random.random()
            g2rand = random.random()
            g = np.array([g1rand, g2rand])
            v1rand = random.random()
            v2rand = random.random()
            v = np.array([v1rand, v2rand])
            gamma_ = 1 - self.lambda_
            # Инициализация списка с количеством итераций
            while (delta > self.epsilon) or (self.max_iter == iteration):
                # Добавление номера итерации
                iteration += 1
                # Подстановка значений W в функцию частных производных
                grad_w1 = self.gradient_w1.subs([(self.w1, w[0])])
                grad_w2 = self.gradient_w2.subs([(self.w2, w[1])])
                # Осуществление градиентного шага
                v[0] = gamma_ * v[0] + (1 - gamma_) * grad_w1
                v[1] = gamma_ * v[1] + (1 - gamma_) * grad_w2
                g[0] = self.alpha * g[0] + (1 - self.alpha) * grad_w1 * grad_w1
                g[1] = self.alpha * g[1] + (1 - self.alpha) * grad_w2 * grad_w2
                v1 = v[0] / (1 - gamma_ ** iteration)
                v2 = v[1] / (1 - gamma_ ** iteration)
                g1 = g[0] / (1 - self.alpha ** iteration)
                g2 = g[1] / (1 - self.alpha ** iteration)
                w1new = w[0] - self.rate * v1 / (sqrt(g1) + self.epsilon)
                w2new = w[1] - self.rate * v2 / (sqrt(g2) + self.epsilon)
                # Расчет разницы между значением на нынешнем и предыдущем шаге
                delta1 = abs(w1new - w[0])
                delta2 = abs(w2new - w[1])
                delta = delta1 if delta1 > delta2 else delta2
                # Обновление значения W, G и V
                w[0] = w1new
                w[1] = w2new
            # Вывод результатов в зависимости от того, удалось найти точку экстремума или нет
            if delta < self.epsilon:
                print("Результаты вычислений по методу Adam: ")
                print("w1 = ", w[0], "w2 = ", w[1])
                print("Количество итераций по методу Adam:", iteration)
            else:
                print("Не удалось найти искомые точки по методу Adam")
            # Добавления значений w и количества итераций в массив
            array_iteration[i] = iteration
            array_w1[i] = w[0]
            array_w2[i] = w[1]

        return array_iteration, array_w1, array_w2
