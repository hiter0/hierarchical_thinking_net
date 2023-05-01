import numpy as np
from itertools import permutations

def permute_matrix(matrix, permutation):
    """根据给定的置换对矩阵进行置换"""
    temp = matrix[permutation]
    return temp[:, permutation]

def upper_tri_sum_sq(matrix):
    """计算矩阵上三角元素的平方和"""
    return np.sum(np.square(np.triu(matrix)))

def find_optimal_permutations(matrix):
    """找到使得上三角元素平方和最大的所有置换"""
    n = matrix.shape[0]
    max_sum_sq = 0
    optimal_perms = []

    # 遍历所有可能的置换
    for perm in permutations(range(n)):
        perm = list(perm)
        perm_matrix = permute_matrix(matrix, perm)
        sum_sq = upper_tri_sum_sq(perm_matrix)

        if sum_sq > max_sum_sq:
            # 发现一个更好的置换，更新最大平方和和最优置换列表
            max_sum_sq = sum_sq
            optimal_perms = [perm]
        elif sum_sq == max_sum_sq:
            # 发现一个同样好的置换，添加到最优置换列表中
            optimal_perms.append(perm)

    # 返回最优置换和对应的置换后的矩阵
    return [(perm, permute_matrix(matrix, perm)) for perm in optimal_perms]

# matrix = np.array([[1, 2, 3],
#                    [4, 5, 6],
#                    [7, 8, 9]])

matrix = np.array([[134, 28,  21,  4,   5,   2],
                   [13,  27,  2,   0,   0,   0],
                   [7 ,  1,   21,  0,   0,   0],
                   [11,  0,   0,   16,  0,   1],
                   [2,   0,   0,   0,   11,  2],
                   [5,   0,   0,   0,   0,   8]])
optimal_permutations = find_optimal_permutations(matrix)

for perm, perm_matrix in optimal_permutations:
    print(f"Permutation: {perm}")
    print(f"Permutated matrix:\n{perm_matrix}")
