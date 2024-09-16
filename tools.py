import os
import sys
from tqdm import tqdm
from colorama import Fore

import numpy as np
from numpy import array, arange, linspace
from numpy import nan, isnan, inf, isinf, pi
from numpy import sin, cos, tan, arctan as atan, sqrt, matmul, resize
from math import radians, degrees, ceil, floor, log10, gcd

from scipy import interpolate, integrate

import multiprocessing as mp
import threading as th

import pandas as pd
import polars as pl

import matplotlib.pyplot as plt
import seaborn as sns

import time

from decorators import timeit


def find(name: str, path: str = os.getcwd()) -> list[str]:
    """Поиск файла/директории по пути"""
    is_file = True if '.' in name else False
    result = list()
    for root, dirs, files in os.walk(path):
        if not is_file:
            for dir in dirs:
                if name in dir:
                    result.append(os.path.join(root, dir))
        for file in files:
            if name in file:
                result.append(os.path.join(root, file))
    return result


def clear_dir(dir: str):
    """Удаление файлов и директории в указанной директории"""
    assert isinstance(dir, str), f'"{dir}" is  not str'
    assert os.path.exists(dir), f'"{dir}" not exists'
    assert os.path.isdir(dir), f'"{dir}" is nor a directory'

    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        try:
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                clear_dir(path)
                os.rmdir(path)
            else:
                raise Exception(f'"{path}" is not a file or a directory')
        except Exception as exception:
            print(f"Failed to delete {path}. Reason: {exception}")


def derivative(f, x0: int | float, method: str = 'central', dx: float = 1e-6) -> float:
    """
    Производная функции f в точке x0

    Parameters
    ----------
    f : function
        Vectorized function of one variable
    x0 : number
        Compute derivative at x = a
    method : string
        Difference formula:
        'central': (f(a+h) - f(a-h)) / 2h
        'forward': (f(a+h) - f(a)) / h
        'backward': (f(a) - f(a-h)) / h

    dx : number
        Step size in difference formula
    """
    assert isinstance(dx, float) and dx != 0

    if method == 'central':
        return (f(x0 + dx / 2) - f(x0 - dx / 2)) / dx
    elif method == 'forward':
        return (f(x0 + dx) - f(x0)) / dx
    elif method == 'backward':
        return (f(x0) - f(x0 - dx)) / dx
    else:
        raise ValueError('method must be "central", "forward" or "backward"!')


def smoothing(x, y, deg: int):
    """Сглаживание кривой полиномом степени deg"""
    p = np.poly1d(np.polyfit(x, y, deg))
    y_smooth = p(x)
    return y_smooth


def distance(p1: tuple, p2: tuple) -> float:
    """Декартово расстояние между 2D точками"""
    assert isinstance(p1, (tuple, list)) and isinstance(p2, (tuple, list))
    assert len(p1) == 2 and len(p2) == 2
    return float(np.linalg.norm(array(p1) - array(p2)))  # sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def distance2line(point: tuple, ABC: tuple) -> float:
    """Расстояние от точки до прямой"""
    assert isinstance(point, (tuple, list))
    assert isinstance(ABC, (tuple, list))
    assert len(point) == 2
    assert len(ABC) == 3
    return abs(ABC[0] * point[0] + ABC[1] * point[1] + ABC[2]) / sqrt(ABC[0] ** 2 + ABC[1] ** 2)


def line_coefficients(func=None, x0=None, p1=None, p2=None) -> tuple[float, float, float]:
    """Коэффициенты A, B, C касательной в точке x0 кривой f или прямой, проходящей через точки p1 и p2"""
    if func is not None and x0 is not None:
        df_dx = derivative(func, x0)
        return df_dx, -1, func(x0) - df_dx * x0
    elif p1 is not None and p2 is not None:
        return (p2[1] - p1[1]) / (p2[0] - p1[0]), -1, (p2[0] * p1[1] - p1[0] * p2[1]) / (p2[0] - p1[0])
    else:
        raise ValueError('func, x0, p1, p2 must not be None!')


def cot(x: float | int) -> float:
    """Котангенс"""
    return 1 / tan(x) if tan(x) != 0 else inf


def tan2cos(tg):
    """Преобразование тангенса в косинус"""
    return sqrt(1 / (tg ** 2 + 1))


# TODO
def av(f, *borders) -> float:
    """Среднее интегральное значение"""
    if len(borders) == 1:
        if borders[0][0] != borders[0][1]:
            return integrate.quad(lambda x1: f(x1), borders[0][0], borders[0][1])[0] / (borders[0][1] - borders[0][0])
        else:
            return f(borders[0][0])
    elif len(borders) == 2:
        if borders[0][0] != borders[0][1] and borders[1][0] != borders[1][1]:
            return (integrate.dblquad(lambda x1, x2: f(x1, x2),
                                      borders[1][0], borders[1][1],
                                      borders[0][0], borders[0][1])[0] /
                    (borders[0][1] - borders[0][0]) /
                    (borders[1][1] - borders[1][0]))
    elif len(borders) == 3:
        pass
    else:
        exit()


# TODO
def av3D(F, T1, T2, P1, P2):
    if T1 != T2 and P1 != P2:
        return integrate.dblquad(lambda T, P: F(T, P), P1, P2, T1, T2)[0] / (T2 - T1) / (P2 - P1)
    elif T1 != T2 and P1 == P2:
        return integrate.quad(lambda T: F(T, P1), T1, T2)[0] / (T2 - T1)
    elif T1 == T2 and P1 != P2:
        return integrate.quad(lambda P: F(T1, P), P1, P2)[0] / (P2 - P1)
    elif T1 == T2 and P1 == P2:
        return F(T1, P1)
    else:
        raise ValueError


def eps(type_eps: str, x1: float | int, x2: float | int) -> float:
    """Погрешность"""
    if type_eps == 'rel':
        try:
            return (x2 - x1) / x1  # относительная
        except ZeroDivisionError:
            return inf
    elif type_eps == 'abs':
        return x2 - x1  # абсолютная
    else:
        raise ValueError('type_eps must be "rel" or "abs"!')  # непонятная


def coprime(a: int, b: int) -> bool:
    """Проверка на взаимно простые числа"""
    return gcd(a, b) == 1


@timeit(4)
def export2(data, file_path='exports', file_name='export_file', file_extension='', show_time=False, **kwargs) -> None:
    """Вывод в файл"""
    assert type(data) == pd.DataFrame or data is plt, 'data must be a DataFrame or matplotlib.pyplot!'
    assert type(file_path) == str, 'type(file_path) must be a string!'
    assert type(file_name) == str, 'type(file_name) must be a string!'
    assert type(file_extension) == str, 'type(file_extension) must be a string!'
    assert file_extension.strip(), 'Empty file_extension!'
    EXTENSIONS = ('txt', 'csv', 'xlsx', 'png', 'pdf', 'svg', 'pkl')
    assert file_extension in EXTENSIONS, f'Unknown extension {file_extension}!'
    file_extension = file_extension.strip().lower()

    os.makedirs(file_path, exist_ok=True)
    file_path = os.getcwd() + '/' + file_path

    сtime = ' ' + time.strftime('%y-%m-%d-%H-%M-%S', time.localtime()) if show_time is True else ''

    print(Fore.YELLOW + f'"{file_path}/{file_name}{сtime}.{file_extension}" file exporting', end='')
    if type(data) is pd.DataFrame:
        if file_extension == 'txt':
            data.to_csv(f'{file_path}/{file_name}{сtime}.{file_extension}', header=kwargs.get('header', True),
                        index=False)
        elif file_extension == 'csv':
            data.to_csv(f'{file_path}/{file_name}{сtime}.{file_extension}', header=kwargs.get('header', True),
                        index=False)
        elif file_extension == 'xlsx':
            data.to_excel(f'{file_path}/{file_name}{сtime}.{file_extension}',
                          sheet_name=kwargs.get('sheet_name', '1'),
                          header=kwargs.get('header', True), index=kwargs.get('index', False))
        elif file_extension.lower() == 'pkl':
            data.to_pickle(f'{file_path}/{file_name}{сtime}.{file_extension.lower()}')
        else:
            print(Fore.RED + 'No such file extension!')
    elif data is plt:
        if file_extension in ('png', 'pdf', 'svg'):
            data.savefig(f'{file_path}/{file_name}{сtime}.{file_extension}',
                         dpi=kwargs.get('dpi', 300),  # разрешение на дюйм
                         bbox_inches='tight',  # обрезать по максимуму
                         transparent=False)  # прозрачность

    else:
        print(f'{Fore.RED}File type {file_extension} did not find!')

    print(Fore.GREEN + f'\r"{file_path}/{file_name}{сtime}.{file_extension}" file has created!')


def angle(k1=nan, k2=nan, points=((), (), ())) -> float:
    """Находит острый угол [рад] между прямыми"""
    if all(points):
        p0, p1, p2 = points  # разархивирование точек
        k1 = (p0[1] - p1[1]) / (p0[0] - p1[0]) if (p0[0] - p1[0]) != 0 else inf
        k2 = (p1[1] - p2[1]) / (p1[0] - p2[0]) if (p1[0] - p2[0]) != 0 else inf
    return abs(atan((k2 - k1) / (1 + k1 * k2)))


def isnum(s, type_num: str = 'float') -> bool:
    """Проверка на число"""
    if type(s) is not str: s = str(s)
    if type_num.lower() in ('int', 'integer'):
        try:
            int(s)
        except ValueError:
            return False
        else:
            return True
    elif type_num.lower() in ('float', 'real'):
        try:
            float(s)
        except ValueError:
            return False
        else:
            return True
    else:
        print('type_num = "int" or "float"'), exit()
    # finally: print('check isnum() done')


def isprime(n: int) -> bool:
    """Проверка на простое число"""
    if not isinstance(n, int): return False
    if n < 2: return False
    for i in range(2, int(sqrt(n)) + 1):
        if n % i == 0: return False
    return True


def prime_factorization(n: int, repeat: bool = True) -> list[int]:
    """Разложение на простые множители"""
    result, divisor = list(), 2
    while divisor * divisor <= n:
        if n % divisor == 0:
            result.append(divisor)
            n //= divisor
        else:
            divisor += 1
    if n > 1: result.append(n)
    return result if repeat else list(set(result))


def isiter(i) -> bool:
    """Проверка на итератор"""
    try:
        for _ in i:
            return True
    except:
        return False


# формируем список из всех римских чисел и новых комбинаций
all_roman = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'),
             (90, 'XC'), (50, 'L'), (40, 'XL'),
             (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]


def to_roman(num: int) -> str:
    """Перевод в римскую систему счисления"""
    roman = ''
    while num > 0:
        for i, r in all_roman:
            while num >= i:  # пока наше число больше или равно числу из словаря
                roman += r  # добавляем соответствующую букву в римское число
                num -= i  # вычитаем словарное число из нашего числа
    return roman


def to_dec(rom: str) -> int:
    """Перевод в десятичную систему счисления"""
    dec = 0
    for i, r in all_roman:
        while rom.startswith(r):  # пока римское число начинается буквы из словаря
            dec += i  # увеличиваем десятичное число на соответствующее значение из словаря
            rom = rom[len(r):]  # убираем найденную букву из римского числа
    return dec


def rounding(s, n=0) -> float:
    """Округление до n значащих цифр"""
    if isnum(s) is False:
        print(Fore.RED + 'can not round not a num!')
    else:
        return float('{:g}'.format(float('{:.{p}g}'.format(s, p=n))))


class Axis:

    @staticmethod
    def transform(x: int | float, y: int | float,
                  x0: int | float = 0, y0: int | float = 0, angle: int | float = 0, scale: int | float = 1):
        """Перенос и поворот осей против часовой стрелки"""
        return scale * matmul(array(([cos(angle), sin(angle)],
                                     [-sin(angle), cos(angle)])),
                              array(([[x - x0],
                                      [y - y0]]))).reshape(2)

    @staticmethod
    def mirror(x, y):
        pass


def coordinate_intersection_lines(ABC1: tuple | list | np.ndarray,
                                  ABC2: tuple | list | np.ndarray) -> tuple[float, float]:
    """Точка пересечения прямых с коэффициентами ABC1 = (A1, B1, C1) и ABC2 = (A2, B2, C2)"""
    assert isinstance(ABC1, (tuple, list, np.ndarray)) and isinstance(ABC2, (tuple, list, np.ndarray))
    assert all(isinstance(el, (int, float, np.number)) for el in ABC1)
    assert all(isinstance(el, (int, float, np.number)) for el in ABC2)

    A1, B1, C1 = ABC1
    A2, B2, C2 = ABC2
    x = -(C2 * B1 - C1 * B2) / (A2 * B1 - B2 * A1) if (A2 * B1 - B2 * A1) != 0 else inf
    y = -(C2 * A1 - A2 * C1) / (B2 * A1 - A2 * B1) if (B2 * A1 - A2 * B1) != 0 else inf

    return x, y


def check_brackets(s: str) -> bool:
    """Проверка на верность постановки скобок"""
    temp = list()
    for i in s:
        if i in ('(', '{', '['):
            temp.append(i)
        else:
            if temp[-1] + i in ('()', '[]', '{}'):
                temp = temp[:-1]
            else:
                return False
    return True


def correlation(data, method='pearson', dropna=False, **kwargs):
    assert type(method) is str, 'type(method) is str'
    assert method.strip().lower() in ('pearson', 'kendall', 'spearman'), \
        'method must be "pearson" or "kendall" or "spearman"!'

    if type(data) is pd.DataFrame:
        cor = data[1:].corr(method=method)
    elif type(data) is np.ndarray:
        cor = np.corrcoef(data)

    if dropna:  # удаление всех строк и столбцов, содержащие только nan
        cor.dropna(how='all', axis=0, inplace=True)  # удаление всех строк содержащих только nan
        cor.dropna(how='all', axis=1, inplace=True)  # удаление всех столбцов содержащих только nan
    return cor


def show_correlation(df, show_num=True, cmap='bwr', fmt=4, savefig=False, **kwargs):
    """Построение матрицы корреляции"""

    assert type(fmt) is int, 'type(fmt) is int'
    assert 1 <= fmt, '1 <= fmt'

    cor = correlation(df, **kwargs)
    if cor is None: return
    if not isnum(fmt, type_num='int') or fmt < 0: fmt = 4
    cor = cor.round(fmt)

    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (12, 12)))
    plt.suptitle('Correlation', fontsize=16, fontweight='bold')
    ax.set_aspect('equal')
    im = ax.imshow(cor, interpolation='nearest',
                   cmap=cmap)  # RGB: 'turbo', # blue vs red: 'bwr', # 2side: 'twilight','twilight_shifted'

    clrbar = fig.colorbar(im, orientation='vertical')
    clrbar.ax.tick_params(labelsize=12)

    clrbar.set_label('Color Intensity []', fontsize=14)
    clrbar.set_ticks(linspace(-1, 1, 21))

    ax.set_xticks(range(len(cor.columns)), cor.columns, fontsize=12, rotation=90)
    ax.set_yticks(range(len(cor.columns)), cor.columns, fontsize=12, rotation=0)
    ax.tick_params(top=False, bottom=False, labeltop=False, labelbottom=True)
    plt.grid(False)

    if show_num:
        for row in range(len(cor.columns)):
            for col in range(len(cor.columns)):
                ax.text(row, col, cor.to_numpy()[row, col],
                        ha="center", va="center", color='black', fontsize=12, rotation=45)

    if savefig: export2(plt, file_path='exports/correlation',
                        file_name='correlation', show_time=True, file_extension='png', **kwargs)

    plt.show()


@timeit()
def run_cpu_tasks_in_parallel(*tasks):
    processes = list()
    for task in tasks:
        if len(task) == 1:
            processes.append(mp.Process(target=task[0]))
        elif len(task) == 2:
            processes.append(mp.Process(target=task[0], args=task[1]))
        else:
            raise 'Incorrect values'
    for process in processes: process.start()
    for process in processes: process.join()


@timeit()
def run_thread_tasks_in_parallel(*tasks):
    threads = list()
    for task in tasks:
        if len(task) == 1:
            threads.append(th.Thread(target=task[0]))
        elif len(task) == 2:
            threads.append(th.Thread(target=task[0], args=task[1]))
        else:
            raise 'Incorrect values'
    for thread in threads: thread.start()
    for thread in threads: thread.join()


def test_f1():
    time.sleep(1)
    a = list()
    for i in range(100_000): a.append(i ** 0.5)
    del a


def test_f2():
    time.sleep(2)
    a = list()
    for i in range(100_000): a.append(i ** 0.5)
    del a


def meke_calc(n):
    time.sleep(n)
    return n


def smart_input(message: str = '', error_message: str = '',
                type_input: str = 'str',
                exceptions=[],
                borders=[-inf, +inf], borders_exceptions: list[int] = []):
    while True:
        var = input(message)

        if type_input.lower() in ('str', 'string'):
            try:
                var = str(var)
                if var not in exceptions:
                    if borders[0] <= len(var) <= borders[1] and len(var) not in borders_exceptions:
                        return var
            except:
                pass
            if error_message:
                print(f'{Fore.RED}{error_message}')
            else:
                print(f'{Fore.RED}Incorrect input!',
                      f'Input type must be {type_input}',
                      f'without {exceptions}',
                      f'with length in {borders[0]}..{borders[1]} without {borders_exceptions}!',
                      sep='\n')
            continue

        elif type_input.lower() in ('int', 'integer'):
            try:
                var = int(var)
                if var not in exceptions:
                    if borders[0] <= var <= borders[1] and var not in borders_exceptions:
                        return var
            except:
                pass
            if error_message:
                print(f'{Fore.RED}{error_message}')
            else:
                print(f'{Fore.RED}Incorrect input!',
                      f'Input type must be {type_input}',
                      f'without {exceptions}',
                      f'in {borders[0]}..{borders[1]} without {borders_exceptions}!',
                      sep='\n')
            continue

        elif type_input.lower() in ('float', 'real'):
            try:
                var = float(var)
                if var not in exceptions:
                    if borders[0] <= var <= borders[1] and var not in borders_exceptions:
                        return var
            except:
                pass
            if error_message:
                print(f'{Fore.RED}{error_message}')
            else:
                print(f'{Fore.RED}Incorrect input!',
                      f'Input type must be {type_input}',
                      f'without {exceptions}',
                      f'in {borders[0]}..{borders[1]} without {borders_exceptions}!',
                      sep='\n')
            continue


def input_clever(message: str = ''):
    while True:
        var = input(message)

        if 'list' in var.lower():
            if var.count('(') == 1 and var.count(')') == 1:
                var = var[var.index('('):var.index(')')]
                var = var.split(', ')
                if 1 <= len(var):
                    return var
            print('Examples "arange" input: arange(9.6) or arange(-3.2, 9) or arange(-3.2, 9, 2.3)')

        elif 'arange' in var.lower():
            if var.count('(') == 1 and var.count(')') == 1:
                var = var[var.index('('):var.index(')')]
                var = var.split(', ')
                if 1 <= len(var) <= 3:
                    start = float(var[0]) if 2 <= len(var) <= 3 else 0
                    stop = float(var[1]) if 2 <= len(var) <= 3 else var[0]
                    step = float(var[2]) if len(var) == 3 else 1
                    return list(arange(start, stop, step))
            print('Examples "arange" input: arange(9.6) or arange(-3.2, 9) or arange(-3.2, 9, 2.3)')

        elif 'linspace' in var.lower():
            if var.count('(') == 1 and var.count(')') == 1:
                var = var[var.index('('):var.index(')')]
                var = var.split(', ')
                if len(var) == 3:
                    start = float(var[0])
                    stop = float(var[1])
                    num = int(var[2])
                    return list(linspace(start, stop, num))
            print('Examples "linspace" input: linspace(-3.2, 9, 2)')


if __name__ == '__main__':
    clear_dir('t')
    exit()

    run_cpu_tasks_in_parallel((test_f1,), (test_f2,))
    run_thread_tasks_in_parallel((test_f1,), (test_f2,))

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(meke_calc, [1, 2, 3])
    print(f'results: {results}')

    exit()
    print(smart_input(message='>>>', borders=[0, 7], borders_exceptions=[3]))

    print(isnum(str(nan)))
    print(rounding(-23456734567.8, 2))  # bug
