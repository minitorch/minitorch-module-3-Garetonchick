"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, Any, List

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(a: float, b: float) -> float:
    return a * b

def id(x: float) -> float:
    return x

def add(a: float, b: float) -> float:
    return a + b

def neg(x: float) -> float:
    return -x

def lt(a: float, b: float) -> bool:
    return a < b

def eq(a: float, b: float) -> bool:
    return a == b

def max(a: float, b: float) -> bool:
    return a if a > b else b

def is_close(a: float, b: float, eps: float = 1e-5) -> bool:
    return abs(a - b) < eps

def exp(x: float) -> float:
    return math.exp(x)

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + exp(-x))

def relu(x: float) -> float:
    return max(x, 0.0)

def log(x: float) -> float:
    return math.log(x)

def inv(x: float) -> float:
    return 1.0 / x

def log_back(x: float, grad: float) -> float:
    return grad / x

def inv_back(x: float, grad: float) -> float:
    return -grad / x**2

def relu_back(x: float, grad: float) -> float:
    return grad if x > 0.0 else 0.0

# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable, iterable: Iterable) -> Iterable:
    return (fn(x) for x in iterable)

def zipWith(iterable1: Iterable, iterable2: Iterable) -> Iterable:
    it1 = iterable1.__iter__()
    it2 = iterable2.__iter__()

    def gen():
        try:
            while True:
                a = next(it1)
                b = next(it2)
                yield (a, b)
        except StopIteration:
            pass

    return gen() 

def reduce(fn: Callable, iterable: Iterable) -> Any:
    it = iterable.__iter__()
    try:
        res = next(it)
    except Exception:
        return None
    try:
        while True:
            res = fn(res, next(it))
    except StopIteration:
        pass
    return res 

def negList(l: List) -> List:
    return list(map(lambda x: -x, l))

def addLists(l1: List, l2: List) -> List:
    return [a + b for a, b in zipWith(l1, l2)]

def sum(l: List) -> Any:
    if len(l) == 0:
        return 0
    return reduce(lambda a, b: a + b, l)

def prod(l: List) -> Any:
    return reduce(lambda a, b: a * b, l)