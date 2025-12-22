import numpy as np
import random


class SlimeMouldAlgorithm:
    """
    Класс реализации алгоритма Slime Mould Algorithm (SMA).
    Алгоритм поддерживает оптимизацию непрерывных функций
    в пространстве произвольной размерности с заданными границами.

    pop_size : int Размер популяции (число особей).
    iter_max : int Максимальное число итераций алгоритма.
    """
    def __init__(self, pop_size, iter_max, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.pop_size = pop_size
        self.iter_max = iter_max

    def _ensure_bounds(self, x, lb, ub):
        """
        Ограничивает вектор решения в пределах допустимых границ.
        Все компоненты вектора x, выходящие за пределы [lb, ub],
        принудительно проецируются на границы.

        :param x : np.ndarray Вектор решения.
        :param lb : np.ndarray Нижняя граница.
        :param ub : np.ndarray Верхняя граница.
        :return: np.ndarray Скорректированный вектор решения.
        """
        x = np.minimum(x, ub)
        x = np.maximum(x, lb)
        return x

    def optimize(self, func, lb, ub, dim):
        """
        Выполняет оптимизацию целевой функции с использованием SMA.
        На каждой итерации популяция сортируется по качеству,
        вычисляются веса особей, после чего выполняется переход
        между режимами исследования и эксплуатации.

        :param func : callable Целевая функция, принимающая вектор и возвращающая скаляр.
        :param lb : float or array-like Нижняя граница области поиска.
        :param ub : float or array-like Верхняя граница области поиска.
        :param dim : int Размерность пространства поиска.
        :return: best_x : np.ndarray Лучшее найденное решение.
        :return: best_f : float Значение целевой функции в лучшей точке.
        :return: best_history : list of float История лучших значений функции по итерациям.
        """
        lb = np.array(lb, dtype=float)
        ub = np.array(ub, dtype=float)
        if lb.shape == ():
            lb = np.full(dim, lb)
        if ub.shape == ():
            ub = np.full(dim, ub)

        pop = np.random.uniform(lb, ub, size=(self.pop_size, dim))
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]

        best_history = [best_f]

        for t in range(1, self.iter_max + 1):
            idx_sorted = np.argsort(fitness)
            pop = pop[idx_sorted]
            fitness = fitness[idx_sorted]

            best = pop[0].copy()
            # worst = pop[-1].copy()
            f_best = fitness[0]
            f_worst = fitness[-1]

            eps = 1e-16
            if abs(f_best - f_worst) < eps:
                weights = np.ones(self.pop_size)
            else:
                weights = (f_worst - fitness) / (f_worst - f_best + eps)
                weights = np.clip(weights, 0.0, 1.0)

            z = 0.03 + 0.97 * ((self.iter_max - t) / self.iter_max)

            for i in range(self.pop_size):
                w = weights[i]
                r1 = random.random()
                r2 = random.random()
                j = random.randrange(self.pop_size)
                k = random.randrange(self.pop_size)
                exploit = best + r1 * w * (ub - lb) * (best - pop[i])
                explore = pop[j] + r2 * (pop[k] - pop[i])

                if random.random() < z:
                    new_x = explore
                else:
                    new_x = exploit

                perturb = (np.random.rand(dim) - 0.5) * 2 * (1 - w) * (ub - lb) * 0.1
                new_x = new_x + perturb

                new_x = self._ensure_bounds(new_x, lb, ub)
                pop[i] = new_x

            fitness = np.array([func(ind) for ind in pop])
            cur_best_idx = np.argmin(fitness)
            cur_best_f = fitness[cur_best_idx]
            cur_best_x = pop[cur_best_idx].copy()

            if cur_best_f < best_f:
                best_f = cur_best_f
                best_x = cur_best_x.copy()

            best_history.append(best_f)

        return best_x, best_f, best_history


if __name__ == "__main__":
    """
    Пример использования Slime Mould Algorithm.
    В качестве тестовой функции используется сферическая функция,
    имеющая глобальный минимум в точке x = 0.
    """
    def sphere(x):
        return np.sum(x**2)

    sma = SlimeMouldAlgorithm(pop_size=50, iter_max=300, seed=42)
    dim = 5
    lb = -5.0
    ub = 5.0
    best_x, best_f, hist = sma.optimize(sphere, lb, ub, dim)
    print("best f:", best_f)
    print("best x:", best_x)
