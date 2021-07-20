#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time : 2020/9/25 20:13
# Author: Zheng Shaoxiang
# @Email : zhengsx95@163.com
# Description:
# 一维装箱问题(One-dimensional bin packing problem, 1D-BPP)问题的
# 分支定价算法(Branch and Price, BP)
from instance import Instance
import basicmodel
from searchTree import SearchTree
from collections import namedtuple
import cProfile
Item = namedtuple("Item", "id width demand")

def load_file(file_name):
    items = []
    with open(file_name, 'r') as file:
        for i in range(2):
            line = file.readline()
        capacity, n = list(int(i) for i in line.strip().split('\t'))
        for i in range(2):
            file.readline()
        for i in range(1, n + 1):
            w = int(file.readline().strip())
            items.append(Item(id=i, width=w))
    return items, capacity

if __name__ == '__main__':
    #instance = Instance()  # 读取文件生成1D-BPP实例
    #print("instance", instance)
    # p = basicmodel.BinPacking({item.id: item for item in instance.items}, instance.capacity)
    #m = bp.solve()
    #bp.print_variables()
    #print(m.Runtime, m.objVal)
    # print(f"-" * 60)
    #tree = SearchTree(instance, verbose=True)  # 初始化搜索树
    #res = tree.solve()
    #print(res)
    # cProfile.run('tree.solve()', sort=1)
    instance = Instance("C:/Users/tangchu2/PycharmProjects/1d_cuttingstock_rl/dataset/Scholl/Scholl_3/HARD0.txt")
    tree = SearchTree(instance, verbose=True)  # 初始化搜索树
    res = tree.solve()
    # print(res)
