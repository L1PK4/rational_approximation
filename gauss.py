import numpy as np


def solve(A, eps=10e-12):
    for i in range(len(A)):
        # поиск индекса ведущего элемента в столбце i
        lead_row_index = np.argmax(np.abs(A[i:, i:i + 1])) + i
        # выбор ведущей строки
        lead_row = A[lead_row_index:lead_row_index + 1]

        if abs(lead_row[0][i]) <= eps:
            print("Матрица вырождена")
            return []

        lead_row /= lead_row[0][i]

        A1 = np.delete(A, lead_row_index, axis=0)

        A1 += lead_row * -A1[:, i:i + 1]

        A = np.insert(A1, i, lead_row, axis=0)

    return A[:, len(A)]
