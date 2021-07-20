#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time : 2020/9/25 20:14
# Author: Zheng Shaoxiang
# @Email : zhengsx95@163.com
# Description:
from collections import namedtuple
import random
Item = namedtuple("Item", "id width demand")

class Instance:
    def __init__(self, file_name=None, seed=0):
        self.capacity = None
        self.n = None
        self.items = []
        self.demand = None
        if file_name is not None:
            self.load_file(file_name)

        else:
            random.seed(seed)
            self.capacity = 10
            self.n = 10   # 客户数
            self.demand = [random.randint(1, 5) for i in range(self.n)]
            self.width = [random.randint(1, self.capacity) for i in range(self.n)]
            self.n = sum(self.demand)
            index = 1
            """for i in range(len(self.demand)):
                for j in range(self.demand[i]):
                    self.items.append(Item(id=index, width=self.width[i], demand=self.demand[i]))
                    index+=1"""
            # print(self.items)
            self.items = [Item(id=i + 1, width=random.randint(1, self.capacity), demand=i+2)
                          for i in range(self.n)]
            # print(self.items)
            pass

    def load_file(self, file_name):
        self.items = []
        with open(file_name, 'r') as file:
            self.n = int(file.readline().strip())
            self.capacity = int(file.readline().strip())

            for i in range(1, self.n + 1):
                w = int(file.readline().strip())
                self.items.append(Item(id=i, width=w,demand=1))

    def __repr__(self):
        return f"capacity={self.capacity}\nitems={self.items}"


if __name__ == '__main__':
    pass
