import gurobipy as grb
from gurobipy import GRB
import numpy as np
from collections import namedtuple
import random
from result_validation import *
import time

# 主问题定义
def master_problem(column, vtype, demand_number_array):
    m = grb.Model()
    x = m.addMVar(shape=column.shape[1], lb=0, )
    m.addConstr(lhs=column @ x >= demand_number_array)
    m.setObjective(x.sum(), GRB.MINIMIZE)
    m.optimize()

    if vtype == GRB.CONTINUOUS:
        return np.array(m.getAttr('Pi', m.getConstrs()))
    else:
        return m.objVal, np.array(m.getAttr('X'))


# 将主问题松弛成线性问题
def restricted_lp_master_problem(column, demand_number_array):
    return master_problem(column, GRB.CONTINUOUS, demand_number_array)


# 找到所有的cut pattern后，求解整数优化主问题。
def restricted_ip_master_problem(column, demand_number_array):
    return master_problem(column, GRB.INTEGER, demand_number_array)


# 子問題求解，寻找pattern
def knapsack_subproblem(kk, demand_width_array, demand_number_array, roll_width):
    m = grb.Model()
    x = m.addMVar(shape=kk.shape[0], lb=0, vtype=GRB.INTEGER)  # 整数规划
    m.addConstr(lhs=demand_width_array @ x <= roll_width)
    m.setObjective(- kk @ x, GRB.MINIMIZE)
    # m.setObjective((kk @ x) - 5, GRB.MAXIMIZE)
    m.optimize()
    # new_col = np.array(m.getAttr('X'))
    # check = 1 - sum(kk*new_col) + 4
    flag_new_column = m.objVal < 0  # 判别数
    if flag_new_column:
        new_column = m.getAttr('X')  # 子问题求解，找到新的列
    else:
        new_column = None
    return flag_new_column, new_column

def test(roll_width, roll_price, demand_width_array, demand_number_array, stop_iteration):
    start_time = time.time()
    # 初始pattern定义
    initial_cut_pattern = np.diag(np.floor(roll_width / demand_width_array))
    # print("initial_cut_pattern", initial_cut_pattern)
    flag_new_cut_pattern = True
    new_cut_pattern = None
    cut_pattern = initial_cut_pattern  # 最近的pattern
    s = time.time()
    while flag_new_cut_pattern:
        # 运行超时停止
        if time.time() - s > 1200:
            break
        if new_cut_pattern:
            cut_pattern = np.column_stack((cut_pattern, new_cut_pattern))  # 最新的pattern。按列合并
        kk = restricted_lp_master_problem(cut_pattern, demand_number_array)  # 将主问题松弛成线性问题
        flag_new_cut_pattern, new_cut_pattern = knapsack_subproblem(kk, demand_width_array, demand_number_array,
                                                                    roll_width)
    # 找到所有的cut pattern后，求解整数优化主问题。
    minimal_stock, optimal_number = restricted_ip_master_problem(cut_pattern, demand_number_array)
    end_time = time.time()
    print('result:')
    print(f'minimal_cost: {minimal_stock}')
    # print(f'optimal_number: {(optimal_number)}')
    print(f'require_number: {sum(optimal_number)}')
    print("time: ", end_time - start_time, "s")
    # 生成结果，输出
    cp1 = np.transpose(cut_pattern).tolist()
    all_pattern = cp1
    dict_stock = [0 for i in range(len(cp1))]
    import csv
    # 创建文件对象
    f = open('RESULT_group_' + str(stop_iteration) +"_length"+str(roll_width)+ ".csv", 'w', encoding='utf-8', newline='' "")
    # 基于文件对象构建csv写入对象
    csv_writer = csv.writer(f)
    title = ["roll_width"] + ["roll_price"] + ["stock_len" + str(int(m)) for m in demand_width_array] + ["required"]
    # 构建列表头
    csv_writer.writerow(title)
    for i in range(len(all_pattern)):
        if optimal_number[i] > 0:
            roll_index = dict_stock[i]
            line = [str(roll_width[roll_index])] + [str(roll_price[roll_index])] + [str(int(m)) for m in all_pattern[i]] + [str(optimal_number[i])]
            csv_writer.writerow(line)
            # print("roll width: ", roll_width[roll_index], " pattern: ", all_pattern[i], " amount: ",optimal_number[i])
    csv_writer.writerow(["minimal_cost", str(minimal_stock)])
    csv_writer.writerow(["require_number", str(sum(optimal_number))])
    csv_writer.writerow(["spend time", str(end_time - start_time) + "s"])
    f.close()

# 随机生成测试数据
def generate_data(roll_width, customer_num):
    roll_width = np.array(roll_width)
    roll_price = np.array([5, 9])
    np.random.seed(1)
    demand_width_array = np.random.randint(10, 100, size=(customer_num))
    np.random.seed(2)
    demand_number_array = np.random.randint(10, 100, size=(customer_num))
    return roll_width, roll_price, demand_width_array, demand_number_array

"""def load_file(file_name):
    demand_width_array,demand_number_array = [],[]
    with open(file_name, 'r') as file:
        n = int(file.readline().strip())
        capacity = int(file.readline().strip())
        for i in range(1, n + 1):
            a = file.readline().strip('\n').split('\t')
            #width, num = list(int(i) for i in file.readline().strip('\n').split('\t'))
            width, num = int(a[0]), int(a[-1])
            demand_width_array.append(width)
            demand_number_array.append(num)
    roll_width = np.array(capacity)
    demand_width_array, demand_number_array = np.array(demand_width_array), np.array(demand_number_array)
    return roll_width, demand_width_array, demand_number_array"""
# No5_factory.csv
# 3509
def load_file(file_name, group):
    demand_width_array, demand_number_array = [], []
    with open(file_name, encoding='utf-8') as file:
        file.readline()
        for i in range(3633):
            a = file.readline()
            if a[0] == str(group):
                a = a.strip('\n').split(',')
                width, num = int(a[2]), int(a[3])
                demand_width_array.append(width)
                demand_number_array.append(num)

    demand_width_array, demand_number_array = np.array(demand_width_array), np.array(demand_number_array)
    return demand_width_array, demand_number_array

# 随机数据集测试
def random_test(customer_count):
    """测试数据生成"""
    roll_width, roll_price, demand_width_array, demand_number_array = generate_data(roll_width=[], customer_num = customer_count)
    """数据测试"""
    test(roll_width, roll_price, demand_width_array, demand_number_array, "random_test")
    """保存 instance 文件"""
    instance = open("true_instance_random_test"+"_rw"+str(roll_width[0]/1000)+".txt", "w")
    write_string = ' '.join(["count: ", str(len(demand_width_array))])
    instance.write(write_string + '\n')
    write_string = ' '.join(["roll width: ", str(roll_width)])
    instance.write(write_string + '\n')
    write_string = ' '.join(["roll_price: ", str(roll_price)])
    instance.write(write_string + '\n')
    write_string = ' '.join(["demand_width_array: ", str(demand_width_array)])
    instance.write(write_string + '\n')
    write_string = ' '.join(["demand_number_array: ", str(demand_number_array)])
    instance.write(write_string + '\n')

def reverse_optimization():
    # 不同的原材料长度，测试单个： roll_width_list = np.array([9600])
    roll_width_list = np.array([i for i in range(9600, 12000, 100)])  # 每隔100mm建立一个。 12m->1200cm->12000mm
    roll_price = np.array([5])
    """ 8个不同类型的真实测试集，杆件编号1-x,2-x...8-x
        测试单个： group_list = np.array([1])"""
    group_list = [i for i in range(1, 8)]
    for group in group_list:
        print(group)
        demand_width_array, demand_number_array = load_file("No5_factory.csv", group)
        for roll_width in roll_width_list:
            roll_width = np.array([roll_width])
            """数据测试"""
            test(roll_width, roll_price, demand_width_array, demand_number_array, stop_iteration=group)
            """检测输出的解是否合法"""
            path = 'RESULT_group_' + str(group) +"_length" + str( roll_width) + ".csv"
            result_validate(path, demand_width_array, demand_number_array)
            """保存 instance 文件"""
            instance = open("true_instance_group_" + str(group) + "_rw" + str(roll_width[0] / 1000) +"_length" + str(
                roll_width)+ ".txt", "w")
            write_string = ' '.join(["count: ", str(len(demand_width_array))])
            instance.write(write_string + '\n')
            write_string = ' '.join(["roll width: ", str(roll_width)])
            instance.write(write_string + '\n')
            write_string = ' '.join(["roll_price: ", str(roll_price)])
            instance.write(write_string + '\n')
            write_string = ' '.join(["demand_width_array: ", str(demand_width_array.tolist())])
            instance.write(write_string + '\n')
            write_string = ' '.join(["demand_number_array: ", str(demand_number_array.tolist())])
            instance.write(write_string + '\n')

reverse_optimization()