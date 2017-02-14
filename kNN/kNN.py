# -*- coding:utf-8 -*-
"""
@author: yangmqglobe
@file: kNN.py
@time: 2017/2/11
"""
import numpy as np
import operator


def create_data_set():
    """
    构造一个用于测试的数据集
    :return: 数据集的矩阵和类别
    """
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def knn_classify(x, data_set, labels, k):
    """
    使用k邻近算法对输入数据进行分类
    :param x: 输入数据
    :param data_set: 训练数据集矩阵
    :param labels: 训练数据集类别
    :param k: k值
    :return: 输入数据所属类别
    """
    data_set_size = data_set.shape[0]
    # 计算输入数据与训练数据集的距离
    diff_mat = np.tile(x, (data_set_size, 1)) - data_set
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    # 对距离结算结果进行排序
    sorted_distances = distances.argsort()
    # 计算距离最近的k个数据的所属分类个数
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_distances[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    # 返回被判定分类概率最大的一个
    return sorted_class_count[0][0]


def auto_norm(data_set):
    """
    对数据集进行归一化
    :param data_set:数据集
    :return: 归一化的数据集， 取值范围以及最小值
    """
    min_vals = data_set.min(0)
    max_vals = data_set.max(0)
    ranges = max_vals - min_vals
    m = data_set.shape[0]
    norm_data_set = data_set - np.tile(min_vals, (m, 1))
    norm_data_set /= np.tile(ranges, (m, 1))
    return norm_data_set, ranges, min_vals
