# -*- coding:utf-8 -*-
"""
@author: yangmqglobe
@file: handwriting.py
@time: 2017/2/11
"""
from .kNN import knn_classify
import numpy as np
import glob
import re


def img2vector(path):
    """
    将一张图片的矩阵读入并生成numpy数组
    :param path: 图片文件路径
    :return:图片数组
    """
    vector = np.zeros((1, 1024))
    with open(path) as f:
        for x, line in enumerate(f):
            line = line.rstrip()
            for y, col in enumerate(line):
                vector[0, 32*x+y] = int(col)
    return vector


def get_data_set(path):
    """
    把文件夹中的所有图片读入生成数据集矩阵
    :param path: 文件路径
    :return: 数据集矩阵和分类
    """
    num_re = re.compile(r'(\d)_\d+\.txt')
    paths = glob.glob('{}/*'.format(path))
    mat = np.zeros((len(paths), 1024))
    labels = []
    for i, path in enumerate(paths):
        mat[i, :] = img2vector(path)
        labels.append(int(num_re.findall(path)[0]))
    return mat, labels


def handwriting_class_test(train_path, test_path, k=3):
    """
    以给定的目录作为训练和测试数据集，对分类进行测试
    :param train_path: 训练数据集
    :param test_path: 测试数据集
    :param k: k值
    """
    hw_mat, hw_labels = get_data_set(train_path)
    paths = glob.glob('{}/*'.format(test_path))
    num_re = re.compile(r'(\d)_\d+\.txt')
    error_count = 0
    test_count = len(paths)
    for i, path in enumerate(paths):
        test_mat = img2vector(path)
        class_result = knn_classify(test_mat, hw_mat, hw_labels, k)
        real_result = int(num_re.findall(path)[0])

        if class_result != real_result:
            error_count += 1
            print('class {} as {} ×'.format(path, class_result))
        else:
            print('class {} as {} √'.format(path, class_result))
    print('total error count: {}'.format(error_count))
    print('total error rate: {:.3f}%'.format(error_count/test_count*100))


def classify_handwring(path, train_path, k=3):
    # todo
    pass
