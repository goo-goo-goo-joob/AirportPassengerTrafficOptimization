import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from tqdm import tqdm

from utils import constraint_cost, calc_time, calc_cost, ConstraintTime, f_ey, f_conv, f_ideal, \
    calc_ideal, calc_conv, fit_ey


class OptimalFinder:
    def _preprocess_bounds(self):
        """
        Определение узлов, в которых не требуется минимизация.
        """
        for j, (m, mm) in enumerate(zip(self.n_min, self.n_max)):
            if m == mm:
                self.idx[j] = m

    def _fit_system(self, plot=True):
        """
        Вычисляет коэффициенты аппроксимирующых полиномов
        и минимальное число каналов обслуживания для каждого узла
        по характеристикам сети.
        :param plot: Визуализировать приближение полиномом в каждом узле.
        :return: Массив массивов коэффициентов каждого узла,
        массив минимального числа каналов каждого узла.
        """
        mu, lam, p = self.params
        for m, l, n in zip(mu, np.insert(lam * np.cumprod(p), 0, lam), self.n_max):
            alpha = l / m
            n_m = int(alpha) + 1
            assert n_m <= n, 'Не выполняется условие существования предельного распределения'
            self.coefs.append(fit_ey(m, l, n_m, n, plot=plot))
            self.n_min.append(n_m)
        self.coefs = np.array(self.coefs, dtype=object)
        self.n_min = np.array(self.n_min)

    def __init__(self, n_max: np.ndarray,
                 s_max: float,
                 t_max: float,
                 s: np.ndarray,
                 params: list,
                 tol=1e-5,
                 near=3,
                 plot=True):
        """
        Инициализация класса оптимизации.
        Вычисление дополнительных параметров.
        :param n_max: Максимально возможное число каналов обслуживания в узле.
        :param s_max: Максимальное значение затрат на каналы обслуживания.
        :param t_max: Максимальное время ожидания в очереди.
        :param s: Массив. Стоимость каждого канала обслуживания.
        :param params[0]: Массив из интенсивностей обслуживания каждого узла сети.
        :param params[1]: Интенсивность входящего потока в первый узел сети.
        :param params[2]: Массив из вероятностей перехода в следующие после первого узлы сети.
        :param tol: Точность соблюдения ограничения.
        :param near: Число ближайших соседей для поиска оптимального решения.
        :param plot: Визуализировать приближение полиномом в каждом узле.
        """
        self.coefs = []
        self.n_min = []
        self.idx = {}
        self.experiments = []
        self.n_max = np.array(n_max)
        self.s_max = s_max
        self.t_max = t_max
        self.s = np.array(s)
        self.params = params
        self.tol = tol
        self.near = near
        self._fit_system(plot=plot)
        self._preprocess_bounds()
        self.n_min_ = self.n_min.copy()
        self.n_max_ = self.n_max.copy()
        self.s_ = self.s.copy()
        self.t_max_ = self.t_max
        self.s_max_ = self.s_max
        if len(self.idx):
            self.t_max -= calc_time(self.n_min[list(self.idx.keys())], list(self.coefs[list(self.idx.keys())]))
            self.s_max -= sum(np.array(list(self.idx.values())) * s[list(self.idx.keys())])
            self.s = np.delete(s, list(self.idx.keys()))
            self.coefs = np.delete(self.coefs, list(self.idx.keys()))
            self.n_min = np.delete(self.n_min, list(self.idx.keys()))
            self.n_max = np.delete(self.n_max, list(self.idx.keys()))
        self.constr_bnds = LinearConstraint(np.eye(len(self.n_max)), self.n_min, self.n_max)
        self.constr_cost = LinearConstraint(self.s, 0, self.s_max)
        self.constraint_time = ConstraintTime(self.t_max, self.coefs)
        self.constr_time = NonlinearConstraint(self.constraint_time, 0, np.inf)
        assert all(self.n_max >= self.n_min), 'Unreachable boundary conditions'
        assert constraint_cost(self.n_min, [self.s_max, self.s]) > -self.tol, 'Constraint unreachable'
        assert self.constraint_time(self.n_max) > -self.tol, 'Constraint unreachable'

    def _check_constraints(self):
        """
        Обработка случая, когда минимизация не требуется.
        """
        if len(self.idx) == len(self.n_min):
            return list(self.idx.values()), list(self.idx.values())

    def _prepare_output(self, result, optimal):
        """
        Подготовка ответа. Добавление узлов, которым не требовалась минимизация.
        """
        res_x = result.x
        for k, v in self.idx.items():
            res_x = np.insert(res_x, k, v)
            optimal = np.insert(optimal, k, v)
        return optimal, res_x

    def _prepare_minimize(self, method):
        """
        Подготовка параметров для функции минимизации minimize.
        """
        assert method in ['SLSQP', 'trust-constr', 'COBYLA'], 'Unsupported method'
        parameters = {
            'method': method,
            'constraints': [self.constr_cost, self.constr_time, self.constr_bnds],
            'options': {'maxiter': 10_000},
            'hess': None,
            'args': (),
        }
        if method == 'trust-constr':
            parameters['options']['initial_barrier_parameter'] = 1e-5
            parameters['options']['initial_barrier_tolerance'] = 1e-5
        return parameters

    def _optimal2int(self, optimal: list, func: callable, bounds, args: list):
        """
        Получает целочисленное решение, которое дает минимальное значение заданной функции
        при выподнении ограничений и граничных условий.
        :param optimal: Не целочисленное решение.
        :param func: Функция расчета критерия.
        :param bounds: Граничные условия решения.
        :param args: Параметры функции критерия.
        :return: Целочисленное решение.
        """
        solutions = []
        for opt, lb, ub in zip(optimal, bounds.lb, bounds.ub):
            int_opt = int(opt)
            optimals = []
            for i in range(self.near):
                if lb <= int_opt - i <= ub:
                    optimals.append(int_opt - i)
                if lb <= int_opt + i + 1 <= ub:
                    optimals.append(int_opt + i + 1)
            solutions.append(optimals)
        solutions = np.array(list(itertools.product(*solutions)))
        if self.t_max:
            ls = np.array([f_ey(x, *self.params) <= self.t_max + self.tol for x in solutions])
            solutions = solutions[ls.ravel()]
        if self.s_max:
            ls = np.array([calc_cost(x, self.s) <= self.s_max + self.tol for x in solutions])
            solutions = solutions[ls.ravel()]
        assert len(solutions), 'Constraints are not met'
        f = np.array([func(x, *args) for x in solutions])
        return solutions[np.argmin(f)]

    def _save_experiment(self, name, method, opt_int, opt):
        """
        Сохранение записи об эксперименте.
        """
        self.experiments.append({
            'Критерий': name,
            'Метод': method,
            'Целое решение,': {str(i + 1): v for i, v in enumerate(opt_int)},
            'Нецелое решение,': {str(i + 1): v for i, v in enumerate(opt)},
            'Среднее время ожидания': f_ey(opt_int, *self.params),
            'Затраты на персонал': calc_cost(opt_int, self.s_),
        })

    def find_optimal_time(self, method: str):
        """
        Метод главного критерия.
        Минимизирует суммарное среднее время ожидания в очереди
        при ограничении на суммарное количество персонала.
        Сохраняет оптимальное решение.
        :param method: Метод нахождения минимума.
        """
        if out := self._check_constraints() is None:
            minimize_kwargs = self._prepare_minimize(method)
            minimize_kwargs['args'] = self.coefs
            res = minimize(calc_time, self.n_max, **minimize_kwargs)
            assert res.success, res.message
            optimal = self._optimal2int(res.x, f_ey, self.constr_bnds, self.params)
            out = self._prepare_output(res, optimal)
        self._save_experiment('Главный критерий (время ожидания)', method, *out)

    def find_optimal_cost(self, method: str):
        """
        Метод главного критерия.
        Минимизирует затраты на обслуживающий персонал
        при ограничении на суммарное время ожидания в очереди.
        Сохраняет оптимальное решение.
        :param method: Метод нахождения минимума.
        """
        if out := self._check_constraints() is None:
            minimize_kwargs = self._prepare_minimize(method)
            minimize_kwargs['args'] = self.s
            if method == 'trust-constr':
                minimize_kwargs['hess'] = lambda x, v: np.zeros((len(self.n_max), len(self.n_max)))
            res = minimize(calc_cost, self.n_min, **minimize_kwargs)
            assert res.success, res.message
            optimal = self._optimal2int(res.x, calc_cost, self.constr_bnds, [self.s])
            out = self._prepare_output(res, optimal)
        self._save_experiment('Главный критерий (количество персонала)', method, *out)

    def find_optimal_conv(self, method: str, weights: list):
        """
        Метод свертки критериев.
        Минимизирует затраты на обслуживающий персонал
        при ограничении на суммарное время ожидания в очереди.
        Сохраняет оптимальное решение.
        :param method: Метод нахождения минимума.
        :param weights: Коэффициенты свертки для каждого критерия.
        """
        if out := self._check_constraints() is None:
            minimize_kwargs = self._prepare_minimize(method)
            minimize_kwargs['args'] = [weights, self.coefs, self.s, self.n_min, self.n_max]
            res = minimize(calc_conv, self.n_min, **minimize_kwargs)
            assert res.success, res.message
            optimal = self._optimal2int(res.x, f_conv, self.constr_bnds,
                                        self.params + [self.n_min, self.n_max, self.s, weights])
            out = self._prepare_output(res, optimal)
        self._save_experiment('Свертка', method, *out)

    def find_optimal_ideal(self, method: str, weights: list, metric: str):
        """
        Метод идеальной точки.
        Минимизирует затраты на обслуживающий персонал
        при ограничении на суммарное время ожидания в очереди.
        Сохраняет оптимальное решение.
        :param method: Метод нахождения минимума.
        :param weights: Массив весов для каждого критерия.
        :param metric: Строка ('2-norm' или 'inf'). Задает норму для вычисления отклонения
        критерия от идеальной точки.
        """
        if out := self._check_constraints() is None:
            minimize_kwargs = self._prepare_minimize(method)
            minimize_kwargs['args'] = [weights, self.coefs, self.s, self.n_min, self.n_max, metric]
            res = minimize(calc_ideal, self.n_min, **minimize_kwargs)
            assert res.success, res.message
            optimal = self._optimal2int(res.x, f_ideal, self.constr_bnds,
                                        self.params + [self.n_min, self.n_max, self.s, weights, metric], )
            out = self._prepare_output(res, optimal)
        if metric == '2-norm':
            name = 'Евклидова норма'
        else:
            name = 'Чебышевская норма'
        self._save_experiment('Идеальная точка', '{} {}'.format(method, name), *out)

    def brute_force(self, w1: list,
                    w2: list,
                    plot=True):
        """
        Полный перебор целых решений для разных критериев.
        :param w1: Массив. Коэффициенты свертки каждого критерия.
        :param w2: Массив. Веса каждого критерия.
        :return: Результаты эксперимента.
        """
        x_time, x_time_val = None, None
        x_cost, x_cost_val = None, None
        x_conv, x_conv_val = None, None
        x_norm2, x_norm2_val = None, None
        x_inf, x_inf_val = None, None
        mu, lam, p = self.params
        if plot:
            ey_plot = []
            cost_plot = []
            n_plot = []
        for n in tqdm(
                itertools.product(*[list(range(self.n_min_[i], self.n_max_[i] + 1)) for i in range(len(self.n_max_))])):
            ey = f_ey(n, mu, lam, p)
            cost = calc_cost(n, self.s_)
            conv = f_conv(n, mu, lam, p, self.n_min_, self.n_max_, self.s_, w1)
            norm_2 = f_ideal(n, mu, lam, p, self.n_min_, self.n_max_, self.s_, w2, '2-norm')
            norm_inf = f_ideal(n, mu, lam, p, self.n_min_, self.n_max_, self.s_, w2, 'inf')
            if ey <= self.t_max_ and cost <= self.s_max_:
                if x_time is None or ey < x_time:
                    x_time = ey
                    x_time_val = ey, cost, n
                if x_cost is None or cost < x_cost:
                    x_cost = cost
                    x_cost_val = ey, cost, n
                if x_conv is None or conv < x_conv:
                    x_conv = conv
                    x_conv_val = ey, cost, n
                if x_norm2 is None or norm_2 < x_norm2:
                    x_norm2 = norm_2
                    x_norm2_val = ey, cost, n
                if x_inf is None or norm_inf < x_inf:
                    x_inf = norm_inf
                    x_inf_val = ey, cost, n
                if plot:
                    ey_plot.append(ey)
                    cost_plot.append(cost)
                    n_plot.append(np.delete(n, list(self.idx.keys())))
        assert x_time, 'Constraints are not met'
        df = [('Главный критерий (время ожидания)', *x_time_val),
              ('Главный критерий (количество персонала)', *x_cost_val), ('Свертка', *x_conv_val),
              ('Идеальная точка, Евкл. норма', *x_norm2_val), ('Идеальная точка, Чеб. норма', *x_inf_val)]
        df = pd.DataFrame(df, columns=['Критерий', 'Среднее время ожидания, сек.', 'Затраты на персонал', 'Решение'])
        if plot:
            plt.figure(figsize=(13, 8))
            plt.scatter(ey_plot, cost_plot, alpha=0.6, label='Возможные значения критериев')
            plt.scatter([x_time_val[0], x_cost_val[0], x_conv_val[0], x_norm2_val[0], x_inf_val[0]],
                        [x_time_val[1], x_cost_val[1], x_conv_val[1], x_norm2_val[1], x_inf_val[1]],
                        s=45,
                        label='Полученные решения')
            plt.title(
                'Возможные значения критериев. Полученные решения.')
            plt.xlabel('Время ожидания заявки в очереди, сек.')
            plt.ylabel('Затраты на обслуживающий персонал, чел.')
            plt.legend()
            plt.show()

            num = np.delete(np.arange(1, len(self.n_min_) + 1), list(self.idx.keys()))
            if len(num) == 2:
                n1_plot, n2_plot = zip(*n_plot)
                fig = px.scatter_3d(
                    pd.DataFrame({f'Кол-во каналов в узле {num[0]}': n1_plot,
                                  f'Кол-во каналов в узле {num[1]}': n2_plot,
                                  'Время ожидания, сек.': ey_plot}),
                    x=f'Кол-во каналов в узле {num[0]}', y=f'Кол-во каналов в узле {num[1]}',
                    z='Время ожидания, сек.', color='Время ожидания, сек.')
                fig.show()

                fig = px.scatter_3d(
                    pd.DataFrame({f'Кол-во каналов в узле {num[0]}': n1_plot,
                                  f'Кол-во каналов в узле {num[1]}': n2_plot,
                                  'Затраты на персонал, чел.': cost_plot}),
                    x=f'Кол-во каналов в узле {num[0]}', y=f'Кол-во каналов в узле {num[1]}',
                    z='Затраты на персонал, чел.', color='Затраты на персонал, чел.')
                fig.show()
            elif len(num) == 3:
                n1_plot, n2_plot, n3_plot = zip(*n_plot)
                fig = px.scatter_3d(
                    pd.DataFrame({f'Кол-во каналов в узле {num[0]}': n1_plot,
                                  f'Кол-во каналов в узле {num[1]}': n2_plot,
                                  f'Кол-во каналов в узле {num[2]}': n3_plot,
                                  'Время ожидания, сек.': ey_plot}),
                    x=f'Кол-во каналов в узле {num[0]}', y=f'Кол-во каналов в узле {num[1]}',
                    z=f'Кол-во каналов в узле {num[2]}', color='Время ожидания, сек.')
                fig.show()

                fig = px.scatter_3d(
                    pd.DataFrame({f'Кол-во каналов в узле {num[0]}': n1_plot,
                                  f'Кол-во каналов в узле {num[1]}': n2_plot,
                                  f'Кол-во каналов в узле {num[2]}': n3_plot,
                                  'Затраты на персонал, чел.': cost_plot}),
                    x=f'Кол-во каналов в узле {num[0]}', y=f'Кол-во каналов в узле {num[1]}',
                    z=f'Кол-во каналов в узле {num[2]}', color='Затраты на персонал, чел.')
                fig.show()
        return df


def main_spb(method):
    """
    Проведение расчетов по предложенной методике для аэропорта Пулково.
    :param method: Метод решения однокритериальной задачи.
    """
    mu1 = [1 / 87, 1 / 30, 1 / 70]
    n_max1 = np.array([88, 26, 24])
    lam1 = 28659.2 / (24 * 60 * 60)
    p1 = [0.999, 0.999]
    s_max = 120
    s = np.ones(len(mu1))
    t_max = 4 * 60
    optf = OptimalFinder(n_max1, s_max, t_max, s, [mu1, lam1, p1])
    optf.find_optimal_time(method)
    optf.find_optimal_cost(method)
    optf.find_optimal_conv(method, [0.8, 0.2])
    for metric in ['2-norm', 'inf']:
        optf.find_optimal_ideal(method, [0.8, 0.2], metric)

    df = pd.json_normalize(optf.experiments, sep=' узел ').sort_values(['Критерий', 'Метод'], ignore_index=True)
    # df.index += 1
    # df.index.name = 'Номер эксперимента'
    df = df.round(2)
    print(df.iloc[:, 2:].to_string(index=False, decimal=','))


def main_svo(method):
    """
    Проведение расчетов по предложенной методике для терминала С аэропорта Шереметьево.
    :param method: Метод решения однокритериальной задачи.
    """
    mu1 = [1 / 75, 1 / 50, 1 / 70]
    n_max1 = np.array([84, 60, 20])
    lam1 = 67307.5 * 0.12 / (24 * 60 * 60)
    p1 = [0.999, 0.999]
    s_max = 40
    s = np.ones(len(mu1))
    t_max = 2 * 60
    optf = OptimalFinder(n_max1, s_max, t_max, s, [mu1, lam1, p1])
    optf.find_optimal_time(method)
    optf.find_optimal_cost(method)
    optf.find_optimal_conv(method, [0.8, 0.2])
    for metric in ['2-norm', 'inf']:
        optf.find_optimal_ideal(method, [0.8, 0.2], metric)

    df = pd.json_normalize(optf.experiments, sep=' узел ').sort_values(['Критерий', 'Метод'], ignore_index=True)
    # df.index += 1
    # df.index.name = 'Номер эксперимента'
    df = df.round(2)
    print(df.iloc[:, 2:].to_string(index=False, decimal=','))


def main_spb_brute_force():
    """
    Проведение расчетов с помощью метода полного перебора для аэропорта Пулково.
    """
    mu1 = [1 / 87, 1 / 30, 1 / 70]
    n_max1 = np.array([88, 26, 24])
    lam1 = 28659.2 / (24 * 60 * 60)
    p1 = [0.999, 0.999]
    s_max = 120
    s = np.ones(len(mu1))
    t_max = 4 * 60
    optf = OptimalFinder(n_max1, s_max, t_max, s, [mu1, lam1, p1])
    n_df = optf.brute_force([0.8, 0.2], [0.8, 0.2])

    n_df = n_df.sort_values(['Критерий'], ignore_index=True)
    n_df = n_df.round(2)
    print(n_df.iloc[:, 1:].to_string(index=False, decimal=','))


def main_svo_brute_force():
    """
    Проведение расчетов с помощью метода полного перебора для терминала С аэропорта Шереметьево.
    """
    mu1 = [1 / 75, 1 / 50, 1 / 70]
    n_max1 = np.array([84, 60, 20])
    lam1 = 67307.5 * 0.12 / (24 * 60 * 60)
    p1 = [0.999, 0.999]
    s_max = 40
    s = np.ones(len(mu1))
    t_max = 2 * 60
    optf = OptimalFinder(n_max1, s_max, t_max, s, [mu1, lam1, p1])
    n_df = optf.brute_force([0.8, 0.2], [0.8, 0.2])

    n_df = n_df.sort_values(['Критерий'], ignore_index=True)
    n_df = n_df.round(2)
    print(n_df.iloc[:, 1:].to_string(index=False, decimal=','))


def main_synthetic(method, size):
    """
    Проведение расчетов по предложенной методике для синтетического примера.
    :param method: Метод решения однокритериальной задачи.
    :param size: Размер сети аэропорта.
    """
    mu1 = [1 / 70] * size
    n_max1 = np.array([24] * size)
    lam1 = 7307.5 * 0.12 / (24 * 60 * 60)
    p1 = [0.999] * (size - 1)
    s_max = 40 * size
    s = np.ones(len(mu1))
    t_max = 1.5 * 60 * size
    optf = OptimalFinder(n_max1, s_max, t_max, s, [mu1, lam1, p1])
    optf.find_optimal_time(method)
    optf.find_optimal_cost(method)
    optf.find_optimal_conv(method, [0.8, 0.2])
    for metric in ['2-norm', 'inf']:
        optf.find_optimal_ideal(method, [0.8, 0.2], metric)

    df = pd.json_normalize(optf.experiments, sep=' узел ').sort_values(['Критерий', 'Метод'], ignore_index=True)
    # df.index += 1
    # df.index.name = 'Номер эксперимента'
    df = df.round(2)
    print(df.iloc[:, 2:].to_string(index=False, decimal=','))


def main_synthetic_brute_force(size):
    """
    Проведение расчетов методом полного перебора для синтетического примера.
    :param size: Размер сети аэропорта.
    """
    mu1 = [1 / 70] * size
    n_max1 = np.array([24] * size)
    lam1 = 7307.5 * 0.12 / (24 * 60 * 60)
    p1 = [0.999] * (size - 1)
    s_max = 40 * size
    s = np.ones(len(mu1))
    t_max = 1.5 * 60 * size
    optf = OptimalFinder(n_max1, s_max, t_max, s, [mu1, lam1, p1])
    n_df = optf.brute_force([0.8, 0.2], [0.8, 0.2])

    n_df = n_df.sort_values(['Критерий'], ignore_index=True)
    n_df = n_df.round(2)
    print(n_df.iloc[:, 1:].to_string(index=False, decimal=','))
