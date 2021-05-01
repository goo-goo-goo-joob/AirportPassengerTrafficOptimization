import multiprocessing
import secrets
from getpass import getpass

import numpy as np
from experiment_collection import Experiment, ExperimentCollectionRemote
from numba import njit
from tqdm.auto import tqdm


@njit
def trace1(LAM: float, MU: float, n: int, prob=1.0, T=1000, seed=None):
    """
    Функция вычисляет среднее время ожидания в очереди системы
    и времена ухода клиентов.
    :param LAM: Интенсивность входящего потока.
    :param MU: Интенсивность обслуживания одним каналом.
    :param n: Количество каналов обслуживания.
    :param prob: Вероятность заявки пройти в следующий узел.
    :param T: Общее время моделирования.
    :param seed: random state.
    :return: Cреднее время ожидания в очереди системы
    и времена ухода клиентов.
    """
    np.random.seed(seed)
    wait_time = 0  # Общее время ожидания в очереди системы
    out_time = []  # Времена ухода клиентов
    L = 0  # Текущая длина очереди
    t_out = np.zeros(n)  # Времена освобождения каналов обслуживания
    t = 0  # Текущее время
    k = 0  # Общее число требований, обслужившихся в системе
    t_in = np.random.exponential(1 / LAM)  # Время прихода очередного требования
    while T >= t_in or L:
        t0 = t
        if T >= t_in:  # Если время моделирования еще не вышло
            t = np.min(np.array([t_in, np.min(t_out)]))
        else:
            t = np.min(t_out)
        h = t - t0
        wait_time += L * h

        if t == np.min(t_out) and L > 0:  # Если освободится канал и очередь непустая
            j = np.argmin(t_out)
            t_out[j] += np.random.exponential(1 / MU)  # Отправляем одного человека из очереди на обслуживание
            if np.random.binomial(1, prob):
                out_time.append(t_out[j])
            L -= 1
        elif t == np.min(t_out) and L == 0:  # Если осбоводился канал и очередь пустая
            j = np.argmin(t_out)
            t_out[j] = t_in + np.random.exponential(1 / MU)  # Пришедшее требование сразу отправляем на обслуживание
            if np.random.binomial(1, prob):
                out_time.append(t_out[j])
            t_in += np.random.exponential(1 / LAM)
            k += 1
        else:  # Новое требование пришло раньше, чем освободился канал (все каналы заняты)
            L += 1
            t_in += np.random.exponential(1 / LAM)
            k += 1

    out_time.sort()

    return wait_time / k, np.array(out_time)


@njit
def trace2(LAM, MU, n, prob=1.0, seed=None):
    """
    Функция возвращает среднее время ожидания в очереди системы
    и времена ухода клиентов.
    :param LAM: Времена прихода клиентов.
    :param MU: Интенсивность обслуживания одним каналом.
    :param n: Количество каналов обслуживания.
    :param prob: Вероятность заявки пройти в следующий узел.
    :param seed: random state.
    :return: Cреднее время ожидания в очереди системы
    и времена ухода клиентов.
    """
    np.random.seed(seed)
    wait_time = 0  # Общее время ожидания в очереди системы.
    out_time = []  # Времена ухода клиентов.
    L = 0  # Текущая длина очереди
    k = 1  # Общее число требований, обслужившихся в системе
    t_out = np.zeros(n)  # Времена освобождения каналов обслуживания
    t = 0  # Текущее время
    T = LAM[-1]  # Продолжительность моделирования
    t_in = LAM[0]  # Время прихода очередного требования
    LAM = LAM[1:]

    while T >= t_in or L:
        t0 = t
        if T >= t_in:  # Если время моделирования еще не вышло
            t = np.min(np.array([t_in, np.min(t_out)]))
        else:
            t = np.min(t_out)
        h = t - t0
        wait_time += L * h

        if t == np.min(t_out) and L > 0:  # Если освободился канал обслуживания и очередь непустая
            j = np.argmin(t_out)
            t_out[j] += np.random.exponential(1 / MU)  # Отправляем одного человека из очереди на обслуживание
            if np.random.binomial(1, prob):
                out_time.append(t_out[j])
            L -= 1
        elif t == np.min(t_out) and L == 0:  # Если осбоводился канал обслуживания и очередь пустая
            j = np.argmin(t_out)
            t_out[j] = t_in + np.random.exponential(1 / MU)  # Пришедшее требование сразу отправляем на обслуживание
            if np.random.binomial(1, prob):
                out_time.append(t_out[j])
            if len(LAM):  # Генерируем новое требование
                t_in = LAM[0]
                LAM = LAM[1:]
                k += 1
            else:
                t_in += 1
        else:  # Новое требование пришло раньше, чем освободился канал (все каналы заняты)
            L += 1
            if len(LAM):  # Генерируем новое требование
                t_in = LAM[0]
                LAM = LAM[1:]
                k += 1
            else:
                t_in += 1

    out_time.sort()

    return wait_time / k, np.array(out_time)


def experiment(k):
    mu1 = [1 / 87, 1 / 30, 1 / 70]  # Интенсивности обслуживания в узлах
    lam1 = 28659.2 / (24 * 60 * 60)  # Поток одиночных пассажиров в секунду
    p1 = [0.999, 0.999]  # Вероятности перехода в следующий узел
    t1, out1 = trace1(lam1, mu1[0], 30, p1[0], T=10_000_000, seed=k)
    t2, out2 = trace2(out1, mu1[1], 11, p1[1], seed=k + 1)
    t3, _ = trace2(out2, mu1[2], 24, seed=k + 2)
    return t1 + t2 + t3, k


if __name__ == '__main__':
    tasks = [secrets.randbelow(2 ** 32) for i in range(1000)]
    exps = ExperimentCollectionRemote('experiments.asciishell.ru', 'main', getpass(), True)
    exps = exps.create_namespace('masha_exp_30_11_24_10000000')
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        for t, seed in tqdm(pool.imap(experiment, tasks), total=len(tasks)):
            exp = Experiment('name_{}'.format(seed), params={'seed': seed}, metrics={'time': t})
            exps.add_experiment(exp, ignore_included=True)
    print(exps.get_experiments()['metrics_time'].mean())
