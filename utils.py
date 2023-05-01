import numpy as np
import torch
from torchinfo import summary
import random
from itertools import permutations

def print_model_info(model):
    input_size = (1, 3, 224, 224)
    model_summary = summary(model, input_size=input_size,
                            col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"])
    total_flops = model_summary.total_mult_adds
    total_params = model_summary.total_params
    print(f"Total FLOPs: {total_flops} Total Params: {total_params}")

def set_random_seed(seed_num):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def permute_matrix(matrix, permutation):
    """根据给定的置换对矩阵进行置换"""
    temp = matrix[permutation]
    return temp[:, permutation]

def upper_tri_sum_sq(matrix):
    """计算矩阵上三角元素的平方和"""
    return np.sum(np.square(np.triu(matrix)))

def filter_matrix(matrix, class_to_index):
    keys = [int(k) for k in class_to_index.keys()]
    class_to_index = np.array(keys)

    # 首先删除主对角线上为0的元素所在的行和列
    diag_elements = matrix.diagonal()
    non_zero_indices = np.where(diag_elements != 0)[0]
    matrix = matrix[np.ix_(non_zero_indices, non_zero_indices)]
    return matrix, class_to_index[non_zero_indices]


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
    return [permute_matrix(matrix, perm) for perm in optimal_perms], optimal_perms

def compare_diag(matrix_list):
    n = matrix_list[0].shape[0]  # 假设所有矩阵都是n*n的，并且n相同
    indices = list(range(len(matrix_list)))  # 初始化索引列表
    for i in range(n):
        max_val = max(matrix[i, i] for matrix in matrix_list)  # 找出当前轮次中的最大值
        new_matrix_list = []  # 新的矩阵列表
        new_indices = []  # 新的索引列表
        for matrix, index in zip(matrix_list, indices):
            if matrix[i, i] >= max_val:  # 如果当前矩阵的值不小于最大值
                new_matrix_list.append(matrix)  # 保留当前矩阵
                new_indices.append(index)  # 保留当前索引
        matrix_list = new_matrix_list
        indices = new_indices
        if len(matrix_list) == 1:  # 如果只剩下一个矩阵，直接返回
            return matrix_list[0], indices[0]
    return matrix_list[0], indices[0]  # 如果全部轮次完成后还没有剩下一个矩阵，返回None



if __name__ == '__main__':
    # 示例成本矩阵 (4x6)
    # cost_matrix = np.array([
    #     [4, 1, 3, 5],
    #     [4, 0, 5, 1],
    #     [3, 2, 2, 8],
    #     [1, 4, 9, 6],
    #     [2, 1, 8, 5],
    #     [2, 4, 9, 1],
    #     [3, 0, 10, 8],
    #     [1, 4, 2, 6]
    # ])
    # #
    # cost_tensor = torch.tensor(cost_matrix, dtype=torch.float32)

    matrix = np.array([[134, 28, 21, 4, 5, 2],
                       [13, 27, 2, 0, 0, 0],
                       [7, 1, 21, 0, 0, 0],
                       [11, 0, 0, 16, 0, 1],
                       [2, 0, 0, 0, 11, 2],
                       [5, 0, 0, 0, 0, 8]])

    result, perms = find_optimal_permutations(matrix)

    optimal_matrix, optimal_indice = compare_diag(result)

    print(optimal_matrix)
    print(perms[optimal_indice])
    print('Done!')


