import math

BOUNDS = (-10, 10)  # limits of every variable
VARIABLE_COUNT = 2  # 2 for function b


def fitness(*variables):  # higher is better
    x, y = variables[0], variables[1]
    term_x = x - 10.0 * math.cos(2.0 * math.pi * x)
    term_y = y - 10.0 * math.cos(2.0 * math.pi * y)
    return 20.0 + term_x + term_y