import random

import numpy as np

from Gradient import Gradient


class NAG(Gradient):

    def __init__(self, gradient_w1, gradient_w2, rate, epsilon, max_iter, lambda_):
        self.gradient_w1 = gradient_w1
        self.gradient_w2 = gradient_w2
        self.rate = rate
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.lambda_ = lambda_

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
            print("Исходная данные для метода NAG:")
            print("w1 = ", w1rand, "w2 = ", w2rand)
            delta = 100
            iteration = 0
            v1rand = random.random()
            v2rand = random.random()
            v = np.array([v1rand, v2rand])
            gamma_ = 1 - self.lambda_
            eta = (1 - gamma_) * self.rate
            while (delta > self.epsilon) and (self.max_iter > iteration):
                # Нахождение численного значения функции, подстановка которой осуществляется в ф-ию частных производных
                wdef1 = w[0] - gamma_ * v[0]
                wdef2 = w[1] - gamma_ * v[1]
                # Подстановка значений W в функцию частных производных
                grad_w1 = self.gradient_w1.subs([(self.w1, wdef1)])
                grad_w2 = self.gradient_w2.subs([(self.w2, wdef2)])
                # Осуществление градиентного шага
                v[0] = gamma_ * v[0] + eta * grad_w1
                v[1] = gamma_ * v[1] + eta * grad_w2
                w1new = w[0] - v[0]
                w2new = w[1] - v[1]
                # Расчет разницы между значением на нынешнем и предыдущем шаге
                delta1 = abs(w1new - w[0])
                delta2 = abs(w2new - w[1])
                delta = delta1 if delta1 > delta2 else delta2
                # Обновление значения W и V
                w[0] = w1new
                w[1] = w2new
                # Добавление номера итерации
                iteration += 1
            # Вывод результатов в зависимости от того, удалось найти точку экстремума или нет
            if delta < self.epsilon:
                print("Результаты вычислений по методу NAG: ")
                print("w1 = ", w[0], "w2 = ", w[1])
                print("Количество итераций по методу NAG:", iteration)
            else:
                print("Не удалось найти искомые точки по методу NAG")
            # Добавления значений w и количества итераций в массив
            array_iteration[i] = iteration
            array_w1[i] = w[0]
            array_w2[i] = w[1]

        return array_iteration, array_w1, array_w2
