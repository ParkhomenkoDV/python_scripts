import numpy as np
from numpy import array, arange, linspace
from numpy import nan, isnan, inf, isinf, pi
from numpy import sin, cos, tan, arctan as atan, sqrt, matmul, resize, gcd

from scipy import interpolate, integrate
from scipy.optimize import curve_fit


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


def approximate(function, x, y):
    """Подбор параметров переданной ф-и"""
    assert callable(function)
    assert isinstance(x, (tuple, list, np.ndarray))
    assert isinstance(y, (tuple, list, np.ndarray))
    popt, covar = curve_fit(function, x, y)  # оптимальные парамеры ф-и и ковариация
    return popt


def smoothing(x, y, deg: int):
    """Сглаживание кривой полиномом степени deg"""
    assert isinstance(deg, int)
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


def angle(k1=nan, k2=nan, points=((), (), ())) -> float:
    """Находит острый угол [рад] между прямыми"""
    if all(points):
        p0, p1, p2 = points  # разархивирование точек
        k1 = (p0[1] - p1[1]) / (p0[0] - p1[0]) if (p0[0] - p1[0]) != 0 else inf
        k2 = (p1[1] - p2[1]) / (p1[0] - p2[0]) if (p1[0] - p2[0]) != 0 else inf
    return abs(atan((k2 - k1) / (1 + k1 * k2)))


def cot(x: float | int) -> float:
    """Котангенс"""
    return 1 / tan(x) if tan(x) != 0 else inf


def tan2cos(tg):
    """Преобразование тангенса в косинус"""
    return sqrt(1 / (tg ** 2 + 1))


def cot2sin(ctg):
    """Преобразование котангенса в синус"""
    return sqrt(1 / (ctg ** 2 + 1))


def is_coprime(a: int, b: int) -> bool:
    """Проверка на взаимно простые числа"""
    return gcd(a, b) == 1


def is_prime(n: int) -> bool:
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
