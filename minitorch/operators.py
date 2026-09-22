import math
from typing import Callable, Iterable, List

EPS = 1e-6


def mul(x: float, y: float) -> float:
    """умножаем два числа"""
    return x * y


def id(x: float) -> float:
    """просто возвращаем что дали"""
    return x


def add(x: float, y: float) -> float:
    """складываем два числа"""
    return x + y


def neg(x: float) -> float:
    """меняем знак"""
    return -x


def lt(x: float, y: float) -> float:
    """единица если первое число меньше"""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """единица если числа равны"""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """выбираем большее"""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """сравниваем с допуском из задания"""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """сигмоида без переполнения экспоненты"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def relu(x: float) -> float:
    """отрицательные значения обнуляем"""
    return x if x > 0 else 0.0


def log(x: float) -> float:
    """натуральный логарифм"""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """экспонента"""
    return math.exp(x)


def inv(x: float) -> float:
    """обратное число"""
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    """производная логарифма с входящим градиентом"""
    return d / (x + EPS)


def inv_back(x: float, d: float) -> float:
    """производная обратного числа с входящим градиентом"""
    return -d / (x * x)


def relu_back(x: float, d: float) -> float:
    """градиент проходит только через положительную часть"""
    return d if x > 0 else 0.0


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], List[float]]:
    """сначала задаем функцию, потом передаем ей список"""
    def apply(ls: Iterable[float]) -> List[float]:
        return [fn(x) for x in ls]

    return apply


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], List[float]]:
    """применяем функцию к парам элементов"""
    def apply(ls1: Iterable[float], ls2: Iterable[float]) -> List[float]:
        return [fn(x, y) for x, y in zip(ls1, ls2)]

    return apply


def reduce(
    fn: Callable[[float, float], float], start: float,
) -> Callable[[Iterable[float]], float]:
    """собираем весь список в одно значение"""
    def apply(ls: Iterable[float]) -> float:
        result = start
        for x in ls:
            result = fn(result, x)
        return result

    return apply


def negList(ls: Iterable[float]) -> List[float]:
    """меняем знаки у всего списка"""
    return map(neg)(ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> List[float]:
    """поэлементная сумма двух списков"""
    return zipWith(add)(ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """сумма, для пустого списка будет ноль"""
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    """произведение, для пустого списка будет единица"""
    return reduce(mul, 1.0)(ls)
