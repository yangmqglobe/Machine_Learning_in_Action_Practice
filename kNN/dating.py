# -*- coding:utf-8 -*-
"""
@author: yangmqglobe
@file: dating.py
@time: 2017/2/11
"""
from .kNN import auto_norm
from .kNN import knn_classify
import linecache
import numpy as np


def file2data_set(path):
    """
    从文件中读取数据集
    :param path: 数据集文件所在路径
    :return: 数据集矩阵和分类
    """
    lines = linecache.getlines(path)
    lines_num = len(lines)
    mat = np.zeros((lines_num, 3))
    labels = []
    for i, line in enumerate(lines):
        line = line.rstrip()
        data = line.split('\t')
        mat[i, :] = data[0:3]
        labels.append(int(data[-1]))
    return mat, labels


def dating_class_test(path, ho_ratio=0.1):
    """
    对约会分类器进行测试
    :param path: 数据集所在路径
    :param ho_ratio: 测试数据集占整体数据集的比例
    """
    dating_mat, dating_labels = file2data_set(path)
    norm_mat = auto_norm(dating_mat)[0]
    all_num = norm_mat.shape[0]
    test_num = int(all_num*ho_ratio)
    error_count = 0
    for i in range(test_num):
        classify_result = knn_classify(
            norm_mat[i, :],
            norm_mat[test_num:all_num, :],
            dating_labels[test_num:all_num],
            3
        )
        print('classify result: {}, real answer: {}'.format(classify_result, dating_labels[i]))
        if classify_result != dating_labels[i]:
            error_count += 1
    print('total error rate: {}%'.format(error_count/test_num*100))


def classify_person(path, k=3, game_time=None, ff_mile=None, ice_cream=None):
    """
    利用给出的数据对一个人精行分类
    :param path: 训练数据集文件路径
    :param k: k值
    :param game_time: 用于玩游戏的时间
    :param ff_mile: 飞行里程
    :param ice_cream: 冰激凌消耗量
    """
    results = ['not at all', 'in small doses', 'in large doses']
    if game_time is None:
        game_time = float(input('percentag time spent in playing video game?'))
    if ff_mile is None:
        ff_mile = float(input('frequent flier miles earned per year?'))
    if ice_cream is None:
        ice_cream = float(input('liters of ice cream consumed per year?'))
    dating_mat, dating_labels = file2data_set(path)
    dating_mat, ranges, min_vals = auto_norm(dating_mat)
    person = np.array((game_time, ff_mile, ice_cream))
    person = person - min_vals / ranges
    classify_result = knn_classify(person, dating_mat, dating_labels, k)
    print('you would probably like this person: {}'.format(results[classify_result - 1]))
