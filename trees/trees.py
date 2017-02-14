# -*- coding:utf-8 -*-
"""
@author: yangmqglobe
@file: trees.py
@time: 2017/2/12
"""
from collections import defaultdict
from math import log
from operator import itemgetter


def create_data_set():
    """
    创建一个用于测试的数据集
    :return: 数据集和分类信息
    """
    data_set = [
            [1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']
        ]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


def split_data_set(data_set, axis, value):
    """
    根据给定的维度对数据集进行切割
    :param data_set: 数据集
    :param axis: 维度
    :param value: 数据值
    :return: 切割后的数据集
    """
    temp_data_set = []
    for vect in data_set:
        if vect[axis] == value:
            temp_vect = vect[:axis]
            temp_vect.extend(vect[axis+1:])
            temp_data_set.append(temp_vect)
    return temp_data_set


def calc_shannon_entropy(data_set):
    """
    计算数据集的香农熵
    :param data_set: 数据集
    :return: 香农熵
    """
    entries_num = len(data_set)
    label_count = defaultdict(int)
    for vect in data_set:
        label_count[vect[-1]] += 1
    shannon_entropy = 0
    for v in label_count.values():
        prob = float(v)/entries_num
        shannon_entropy -= prob*log(prob, 2)
    return shannon_entropy


def get_best_split_feature(data_set):
    """
    获得最佳的数据集划分索引值
    :param data_set: 数据集
    :return: 划分索引值
    """
    feature_num = len(data_set[0]) - 1
    base_entropy = calc_shannon_entropy(data_set)
    best_info_gain = 0
    best_split_feature = -1
    for i in range(feature_num):
        featurs = [vect[0] for vect in data_set]
        unique_vals = set(featurs)
        new_entropy = 0
        for value in unique_vals:
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set)/len(data_set)
            new_entropy += prob * calc_shannon_entropy(sub_data_set)
            info_gain = base_entropy - new_entropy
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_split_feature = i
    return best_split_feature


def major_class(class_list):
    """
    计算主要的类标签
    由于划分的极限，当无法再划分数据集时，以多类为准
    :param class_list: 类标签集合
    :return: 主要的类标签
    """
    class_count = defaultdict(int)
    for vout in class_list:
        class_count[vout] += 1
    sorted_class_count = sorted(class_count.items(), key=itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_tree(data_set, labels):
    """
    创建树
    :param data_set: 数据集
    :param labels: 分类标签集
    :return: 决策树
    """
    temp_labels = labels[:]
    class_list = [vect[-1] for vect in data_set]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(data_set[0]) == 1:
        return major_class(class_list)
    best_feature = get_best_split_feature(data_set)
    best_feature_label = temp_labels[best_feature]
    tree = {
        best_feature_label: {}
    }
    del temp_labels[best_feature]
    feature_vals = [vect[best_feature] for vect in data_set]
    unique_vals = set(feature_vals)
    for value in unique_vals:
        sub_labels = temp_labels[:]
        tree[best_feature_label][value] = create_tree(
            split_data_set(data_set, best_feature, value),
            sub_labels
        )
    return tree


def get_leafs_num(tree):
    """
    获得树的叶结点个数，以在绘制树是分配空间
    :param tree: 决策树
    :return: 树的叶结点个数
    """
    if isinstance(tree, dict):
        leafs_num = 0
        root = list(tree.values())[0]
        for node in root.values():
            leafs_num += get_leafs_num(node)
        return leafs_num
    else:
        return 1


def get_tree_deep(tree):
    """
    获得树的深度，以在绘制树的时候分配空间
    :param tree: 决策树
    :return: 树的最大深度
    """
    if isinstance(tree, dict):
        deep = 0
        root = list(tree.values())[0]
        for node in root.values():
            temp_deep = get_tree_deep(node)
            if temp_deep > deep:
                deep = temp_deep
        return deep + 1
    else:
        return 0


def classify(x, labels, tree):
    """
    对给定的数据进行分类
    :param x: 输入数据
    :param labels: 与输入数据顺序对应的分支列表
    :param tree: 决策树
    :return: 分类结果
    """
    if isinstance(tree, dict):
        key, root = list(tree.items())[0]
        branch = x[labels.index(key)]
        return classify(x, labels, root[branch])
    else:
        return tree
