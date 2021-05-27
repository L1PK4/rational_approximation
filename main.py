import numpy as np

import gauss


def f(_x):
    return (_x**3 - 1 * _x**2 + 3 * _x - 5) / (2 * _x**2 - 3 * _x + 7)


if __name__ == "__main__":
    k = 3
    l = 2
    n = k + l + 1

    x = np.array([5, 4, 3, 2, 1, 0]).astype(np.float)

    a = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if j < k:
                a[i][j] = x[i]**j
            else:
                a[i][j] = -x[i] ** (j - k) * f(x[i])

    b = np.array([-_x**k for _x in x])

    A = np.hstack((a, b.reshape((len(b), 1))))

    x = gauss.solve(A)
    print(x)
