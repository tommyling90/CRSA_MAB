def get_Y_space(k):
    Y = set(range(0, k ** 2))
    return Y

def get_U_space(k):
    U = set(range(0, k ** 2))
    U.add("accept")
    return U