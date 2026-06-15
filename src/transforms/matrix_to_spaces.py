def get_Y_space(k):
    Y = set(range(1, k ** 2 + 1))
    return Y

def get_U_space(k):
    U = set(range(1, k ** 2 + 1))
    U.add("accept")
    return U