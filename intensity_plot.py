import calendar
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def transform_data(data):
    time_data = data[month][::-1].stack().reset_index().iloc[:, 2]
    dt = []
    for i, _ in enumerate(time_data):
        dt.append(date(2007 + i // 12, i % 12 + 1, 1))

    new_data = pd.DataFrame({'date': dt, 'data': time_data})
    new_data.index = pd.PeriodIndex(dt, freq='M')
    new_data.index = new_data.index.to_timestamp()
    return new_data


def plot_data(data, name):
    ax = data['data'].plot(figsize=(13, 8),
                           title=f'Пассажиропоток за 2007 - 2021 (по апрель) годы по месяцам {name}')
    ax.set_xlabel("Время")
    ax.set_ylabel("Количество пассажиров в месяц")
    plt.grid()
    plt.show()


def plot_intensity(data, name, year):
    n_year = 366 if calendar.isleap(year) else 365
    m_year = np.array([calendar.monthrange(year, i)[1] for i in range(1, 13)])
    m_year_t = m_year.cumsum()
    x = np.sort(list(np.arange(0, n_year, 1)) + list(m_year_t))
    global flag
    flag = True

    def f_cond(x):
        global flag
        cond = [False] * 12
        m = np.where(x <= m_year_t)[0][0]
        if flag and any(x == m_year_t):
            flag = False
        elif not flag and any(x == m_year_t):
            m += 1
            flag = True
        cond[m] = True
        return 0.8 * data[cond] / (2 * m_year[m])

    plt.figure(figsize=(13, 8))
    plt.plot(x, [f_cond(xi) for xi in x])
    plt.title(f'Интенсивность входящего потока в узел 1 сети, {name}, {year} год')
    plt.xlabel('t, дни')
    plt.ylabel('$\lambda_1$, пасс./день')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    # link = 'http://www.favt.ru/opendata/7714549744-statperevaeroportpas/'
    path = 'data-20210201-structure-20181102.csv'
    df = pd.read_csv(path, encoding='windows-1251', sep=';')
    df.replace('***', 0, inplace=True)

    month = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь', 'Июль', 'Август',
             'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']
    for m in month + ['Январь - Декабрь']:
        df[m] = df[m].apply(lambda x: float(str(x).replace(' ', '')))

    year = 2019
    name_svo = 'Москва(Шереметьево)'
    svo = df[df['Наименование аэропорта РФ'] == name_svo]
    df_svo = transform_data(svo)
    plot_data(df_svo, name_svo)
    plot_intensity(svo.iloc[1, :][month], name_svo, year)

    name_spb = 'Санкт-Петербург(Пулково)'
    spb = df[df['Наименование аэропорта РФ'] == name_spb]
    df_spb = transform_data(spb)
    plot_data(df_spb, name_spb)
    plot_intensity(spb.iloc[1, :][month], name_spb, year)
