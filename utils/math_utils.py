import numpy as np
from math import factorial

def get_max_n(k):
    m = k * k
    fact = factorial(m)

    n = 0
    while (n + 1) ** m < fact:
        n += 1

    return n