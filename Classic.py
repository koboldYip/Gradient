import random

import numpy as np

from Gradient import Gradient


class Classic(Gradient):

    def __init__(self, gradient_w1, gradient_w2, rate, epsilon, max_iter):
        self.gradient_w1 = gradient_w1
        self.gradient_w2 = gradient_w2
        self.rate = rate
        self.epsilon = epsilon
        self.max_iter = max_iter

    # Задание функции для решения градиентным методом
    def process(self):
        array_iteration = np.zeros(100)
        array_w1 = np.zeros(100)
        array_w2 = np.zeros(100)
        for i in range(100):
            w1rand = random.random()
            w2rand = random.random()
            w = np.array([w1rand, w2rand])
            print("Исходная данные для метода Classic:")
            print("w1 = ", w1rand, "w2 = ", w2rand)
            delta = 100
            iteration = 0
            # Инициализация списка с количеством итераций
            while (delta > self.epsilon) and (self.max_iter > iteration):
                # Подстановка значений W в функцию частных производных
                grad_w1 = self.gradient_w1.subs([(self.w1, w[0])])
                grad_w2 = self.gradient_w2.subs([(self.w2, w[1])])
                # Осуществление градиентного шага
                w1new = w[0] - self.rate * grad_w1
                w2new = w[1] - self.rate * grad_w2
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
                print("Результаты вычислений по классическому методу: ")
                print("w1 = ", w[0], "w2 = ", w[1])
                print("Количество итераций по классическому методу:", iteration)
            else:
                print("Не удалось найти искомые точки по классическому методу")
            # Добавления значений w и количества итераций в массив
            array_iteration[i] = iteration
            array_w1[i] = w[0]
            array_w2[i] = w[1]

        return array_iteration, array_w1, array_w2
