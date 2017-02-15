# -*- coding:utf-8 -*-
"""
@author: yangmqglobe
@file: contactlenses.py
@time: 2017/2/14
"""
import linecache


def load_data_set(path):
    lines = linecache.getlines(path)
    data_set = [line.rstrip().split('\t') for line in lines]
    labels = ['age', 'prescript', 'astigmatic', 'teat rate']
    return data_set, labels
