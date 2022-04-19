import random

import numpy as np
from sympy import *

from Gradient import Gradient


class AdaDelta(Gradient):

    def __init__(self, gradient_w1, gradient_w2, alpha, epsilon, max_iter, delta, Delta):
        self.gradient_w1 = gradient_w1
        self.gradient_w2 = gradient_w2
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.delta = delta
        self.Delta = Delta
        self.alpha = alpha

    # Задание функции для решения градиентным методом
    def process(self):
        array_iteration = np.zeros(100)
        array_w1 = np.zeros(100)
        array_w2 = np.zeros(100)
        for i in range(100):
            # Инициализация искомых параметров W случайными числами
            w1rand = random.random()
            w2rand = random.random()
            w = np.array([w1rand, w2rand])
            print("Исходная данные для метода AdaDelta:")
            print("w1 = ", w1rand, "w2 = ", w2rand)
            difference = 100
            iteration = 0
            g1rand = random.random()
            g2rand = random.random()
            g = np.array([g1rand, g2rand])
            delta1 = self.delta
            delta2 = self.delta
            Delta1 = self.Delta
            Delta2 = self.Delta
            # Цикл, работающий до тех пор, пока не выполнятся условия остановки цикла
            while (difference > self.epsilon) and (self.max_iter > iteration):
                # Подстановка значений W в функцию частных производных
                grad_w1 = self.gradient_w1.subs([(self.w1, w[0])])
                grad_w2 = self.gradient_w2.subs([(self.w2, w[1])])
                # Осуществление градиентного шага
                g1new = self.alpha * g[0] + (1 - self.alpha) * grad_w1 * grad_w1
                g2new = self.alpha * g[1] + (1 - self.alpha) * grad_w2 * grad_w2
                delta1 = grad_w1 * (sqrt(Delta1) + self.epsilon) / (sqrt(g1new) + self.epsilon)
                delta2 = grad_w2 * (sqrt(Delta2) + self.epsilon) / (sqrt(g2new) + self.epsilon)
                Delta1 = self.alpha * Delta1 + (1 - self.alpha) * delta1 * delta1
                Delta2 = self.alpha * Delta2 + (1 - self.alpha) * delta2 * delta2
                w1new = w[0] - delta1
                w2new = w[1] - delta2
                # Расчет разницы между значением на нынешнем и предыдущем шаге
                difference1 = abs(w1new - w[0])
                difference2 = abs(w2new - w[1])
                difference = difference1 if difference1 > difference2 else difference2
                # Обновление значения W и G
                w[0] = w1new
                w[1] = w2new
                g[0] = g1new
                g[1] = g2new
                # Добавление номера итерации
                iteration += 1
            # Вывод результатов в зависимости от того, удалось найти точку экстремума или нет
            if difference < self.epsilon:
                print("Результаты вычислений по методу AdaDelta: ")
                print("w1 = ", w[0], "w2 = ", w[1])
                print("Количество итераций по методу AdaDelta:", iteration)
            else:
                print("Не удалось найти искомые точки по методу AdaDelta")
                print("Количество итераций по методу AdaDelta:", iteration)
            # Добавления значений w и количества итераций в массив
            array_iteration[i] = iteration
            array_w1[i] = w[0]
            array_w2[i] = w[1]

        return array_iteration, array_w1, array_w2
