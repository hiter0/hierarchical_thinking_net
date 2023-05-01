import os
import csv
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils import *
def rectify_predict(target_row_number):
    # 存储所有的预测结果
    predictions = []# 存储所有出现过的类别
    # 你感兴趣的样本的行号
    classes = set()
    # 读取所有CSV文件
    weights_dir = 'output/top_k'
    for weights_file in os.listdir(weights_dir):
        if weights_file.endswith('.csv'):
            with open(os.path.join(weights_dir, weights_file), 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for i, row in enumerate(reader):
                    # 只记录你感兴趣的样本的预测结果
                    if i == target_row_number:
                        top1_class = row['top1_class']
                        top2_class = row['top2_class']
                        predictions.append((top1_class, top2_class))
                        classes.add(top1_class)
                        classes.add(top2_class)

    # 创建一个类别到索引的映射
    class_to_index = {cls: index for index, cls in enumerate(sorted(classes))}

    # 创建一个空的混淆矩阵
    confusion_matrix = np.zeros((len(classes), len(classes)), dtype=int)

    # 遍历所有的预测结果
    for top1_class, top2_class in predictions:
        i = class_to_index[top1_class]
        j = class_to_index[top2_class]
        confusion_matrix[i, i] += 1
        confusion_matrix[i, j] += 1

    # 打印混淆矩阵
    print(confusion_matrix)

    confusion_matrix, class_to_index = filter_matrix(confusion_matrix, class_to_index)
    result, perms = find_optimal_permutations(confusion_matrix)
    # print(result)

    optimal_matrix, optimal_indice = compare_diag(result)
    new_order = perms[optimal_indice]
    print(optimal_matrix)
    print(new_order)
    print('len(perms):' + str(len(perms)))

    classes = [class_to_index[i] for i in new_order]
    print(classes)

    return optimal_matrix, classes

def visualize_matrix(matrix, classes, i):

    # 将numpy数组转换为pandas DataFrame，以便于使用seaborn绘图
    optimal_matrix_df = pd.DataFrame(matrix, index=classes, columns=classes)

    # 创建一个网格图。annot=True表示在每个单元格中写入数值。cmap='Blues'表示使用蓝色调色板。
    sns.heatmap(optimal_matrix_df, annot=True, cmap='Blues', fmt='g')

    # 设置标题和轴标签
    title = 'Confusion Matrix Num_' + str(i)
    plt.title(title)

    # 将列的刻度放在矩阵图的上方
    plt.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.xticks(rotation=0)  # 旋转列的刻度标签，使其更易阅读
    plt.yticks(rotation=0)  # 旋转列的刻度标签，使其更易阅读
    plt.xlabel('Predict:TOP2')
    plt.ylabel('Answer:TOP1')
    output_dir = 'confusion_matrix'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f'{output_dir}/{title}.png')
    plt.savefig(f'output/{output_dir}/{title}.png')
    # 显示图形
    # plt.show()
    plt.clf()

def main():
    # sample_index = [i for i in range(0, 212)]
    sample_index = [i for i in range(213, 236)]
    # 计算量异常的编号212
    for i in sample_index:
        rectified_matrix, rectified_classes = rectify_predict(i)
        visualize_matrix(rectified_matrix, rectified_classes, i)

if __name__ == '__main__':
    main()