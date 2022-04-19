import random

import numpy as np
from sympy import *

from Gradient import Gradient


class RMSProp(Gradient):

    def __init__(self, gradient_w1, gradient_w2, epsilon, max_iter, lambda_):
        self.gradient_w1 = gradient_w1
        self.gradient_w2 = gradient_w2
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.lambda_ = lambda_

    # Задание функции для решения градиентным методом
    def process(self):
        array_iteration = np.zeros(100)
        array_w1 = np.zeros(100)
        array_w2 = np.zeros(100)
        # Цикл, запускающий исходный код 100 раз
        for i in range(100):
            # Инициализация искомых параметров W случайными числами
            w1rand = random.random()
            w2rand = random.random()
            w = np.array([w1rand, w2rand])
            print("Исходная данные для метода RMSProp:")
            print("w1 = ", w1rand, "w2 = ", w2rand)
            delta = 100
            iteration = 0
            g1rand = random.random()
            g2rand = random.random()
            g = np.array([g1rand, g2rand])
            gamma_ = 1 - self.lambda_
            while (delta > self.epsilon) and (self.max_iter > iteration):
                # Подстановка значений W в функцию частных производных
                grad_w1 = self.gradient_w1.subs([(self.w1, w[0])])
                grad_w2 = self.gradient_w2.subs([(self.w2, w[1])])
                # Осуществление градиентного шага
                g[0] = gamma_ * g[0] + (1 - gamma_) * grad_w1 * grad_w1
                g[1] = gamma_ * g[1] + (1 - gamma_) * grad_w2 * grad_w2
                w1new = w[0] - (1 - gamma_) * grad_w1 / sqrt(g[0] + self.epsilon)
                w2new = w[1] - (1 - gamma_) * grad_w2 / sqrt(g[1] + self.epsilon)
                # Расчет разницы между значением на нынешнем и предыдущем шаге
                delta1 = abs(w1new - w[0])
                delta2 = abs(w2new - w[1])
                delta = delta1 if delta1 > delta2 else delta2
                # Обновление значения W
                w[0] = w1new
                w[1] = w2new
                # Добавление номера итерации
                iteration += 1
            # Вывод результатов в зависимости от того, удалось найти точку экстремума или нет
            if delta < self.epsilon:
                print("Результаты вычислений по методу RMSProp: ")
                print("w1 = ", w[0], "w2 = ", w[1])
                print("Количество итераций по методу RMSProp:", iteration)
            else:
                print("Не удалось найти искомые точки по методу RMSProp")
                print("Количество итераций по методу RMSProp:", iteration)
            # Добавления значений w и количества итераций в массив
            array_iteration[i] = iteration
            array_w1[i] = w[0]
            array_w2[i] = w[1]

        return array_iteration, array_w1, array_w2
