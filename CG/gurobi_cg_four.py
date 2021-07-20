"""4种不同长度原始钢材的切割问题"""
import gurobipy as grb
import numpy as np
from gurobipy import GRB
import time
from result_validation import *

"""column1, column2, column3, column4分别为四种不同规格钢管的pattern"""
def master_problem(column1, column2, column3, column4, vtype, demand_number_array, roll_price):
    m = grb.Model()
    # x,y,z,k 分别代表四种规格钢管的切割方式
    x = m.addMVar(shape=column1.shape[1], lb=0, vtype=vtype)
    y = m.addMVar(shape=column2.shape[1], lb=0, vtype=vtype)
    z = m.addMVar(shape=column3.shape[1], lb=0, vtype=vtype)
    k = m.addMVar(shape=column4.shape[1], lb=0, vtype=vtype)
    m.addConstr(lhs=column1 @ x + column2 @ y + column3 @ z + column4 @ k >= demand_number_array)
    # 目标函数是四种规格钢管数量的价格总和
    m.setObjective(x.sum()*roll_price[0] + y.sum()*roll_price[1] + z.sum()*roll_price[2] + k.sum()*roll_price[3], GRB.MINIMIZE)
    m.optimize()
    # 将结果保存在本地文件夹
    m.write("C:/Users/tangchu2/PycharmProjects/1d_cuttingstock_rl/CG/out.sol")
    # 如果是松弛后的线性优化，输出π
    if vtype == GRB.CONTINUOUS:
        return np.array(m.getAttr('Pi', m.getConstrs()))
    else:
        # 找到所有的pattern 或者达到终止循环条件，求解整数优化主问题，输出目标值和系数
        return m.objVal, np.array(m.getAttr('X')),

# 将主问题松弛成线性问题求解
def restricted_lp_master_problem(column1, column2, column3, column4, demand_number_array,roll_price):
    return master_problem(column1, column2, column3, column4, GRB.CONTINUOUS, demand_number_array, roll_price)

# 找到所有的cut pattern后，求解整数优化主问题。
def restricted_ip_master_problem(column1, column2, column3, column4, demand_number_array,roll_price):
    return master_problem(column1, column2, column3, column4, GRB.INTEGER, demand_number_array, roll_price)

"""
子问题求解, 寻找最优pattern。
分别求解四个子问题，选择目标函数最小的K个子问题的解
"""
def knapsack_subproblem(para_x, demand_width_array,roll_width,roll_price):
    first_stock_model = grb.Model()
    x = first_stock_model.addMVar(shape=para_x.shape[0], lb=0, vtype=GRB.INTEGER)  # 整数规划
    first_stock_model.addConstr(lhs=demand_width_array @ x <= roll_width[0])
    first_stock_model.setObjective(- para_x @ x+roll_price[0], GRB.MINIMIZE)  # 第一根钢管的价格
    first_stock_model.optimize()

    second_stock_model = grb.Model()
    y = second_stock_model.addMVar(shape=para_x.shape[0], lb=0, vtype=GRB.INTEGER)  # 整数规划
    second_stock_model.addConstr(lhs=demand_width_array @ y <= roll_width[1])
    second_stock_model.setObjective(- para_x @ y + roll_price[1], GRB.MINIMIZE)  # 第二根钢管的价格
    second_stock_model.optimize()

    third_stock_model = grb.Model()
    z = third_stock_model.addMVar(shape=para_x.shape[0], lb=0, vtype=GRB.INTEGER)  # 整数规划
    third_stock_model.addConstr(lhs=demand_width_array @ z <= roll_width[2])
    third_stock_model.setObjective(- para_x @ z + roll_price[2], GRB.MINIMIZE)  # 第三根钢管的价格
    third_stock_model.optimize()

    fourth_stock_model = grb.Model()
    k = fourth_stock_model.addMVar(shape=para_x.shape[0], lb=0, vtype=GRB.INTEGER)  # 整数规划
    fourth_stock_model.addConstr(lhs=demand_width_array @ k <= roll_width[3])
    fourth_stock_model.setObjective(- para_x @ k + roll_price[3], GRB.MINIMIZE)  # 第四根钢管的价格
    fourth_stock_model.optimize()
    judge_list = [first_stock_model.objVal, second_stock_model.objVal, third_stock_model.objVal, fourth_stock_model.objVal]
    min_flag = min(judge_list)
    column_position = 0
    if min_flag < 0:
        if min_flag == first_stock_model.objVal:
            new_column = first_stock_model.getAttr('X')  # 第一个子问题求解，找到新的列
            new_column_price = np.array([roll_price[0]])
            column_position = 1
        elif min_flag == second_stock_model.objVal:
            new_column = second_stock_model.getAttr('X')  # 第二个子问题求解，找到新的列
            new_column_price = np.array([roll_price[1]])
            column_position = 2
        elif min_flag == third_stock_model.objVal:
            new_column = third_stock_model.getAttr('X')  # 第三个子问题求解，找到新的列
            new_column_price = np.array([roll_price[2]])
            column_position = 3
        else:
            new_column = fourth_stock_model.getAttr('X')  # 第四个子问题求解，找到新的列
            new_column_price = np.array([roll_price[3]])
            column_position = 4
    else:  # 找到所有的列
        new_column = None
        new_column_price = None
    flag_new_cut_pattern = True if min_flag < 0 else False
    return flag_new_cut_pattern, new_column, new_column_price, column_position


def test(roll_width, roll_price, demand_width_array, demand_number_array, stop_iteration):
    start_time = time.time()
    # 初始定义最简单的pattern
    # 初始化第一种stock规格的切割方式
    initial_cut_pattern = np.diag(np.floor(roll_width[0] / demand_width_array))
    # 初始化第二种stock规格的切割方式
    initial_cut_pattern_second = np.diag(np.floor(roll_width[1] / demand_width_array))
    # 初始化第三种stock规格的切割方式
    initial_cut_pattern_third = np.diag(np.floor(roll_width[2] / demand_width_array))
    # 初始化第四种stock规格的切割方式
    initial_cut_pattern_fourth = np.diag(np.floor(roll_width[3] / demand_width_array))
    flag_new_cut_pattern = True
    new_cut_pattern = None
    cut_pattern, cut_pattern_second, cut_pattern_third, cut_pattern_fourth = initial_cut_pattern, initial_cut_pattern_second, initial_cut_pattern_third, initial_cut_pattern_fourth  # 最开始的pattern
    stop = 0
    s = time.time()
    while flag_new_cut_pattern:
        #if stop > stop_iteration:
            # break
        # 运行超过2个小时停止
        if time.time() - s > 300:
            break
        if new_cut_pattern:
            if column_position == 1:
                cut_pattern = np.column_stack((cut_pattern, new_cut_pattern))    # 最新的pattern。按列合并
            elif column_position == 2:
                cut_pattern_second = np.column_stack((cut_pattern_second, new_cut_pattern))    # 最新的pattern。按列合并
            elif column_position == 3:
                cut_pattern_third = np.column_stack((cut_pattern_third, new_cut_pattern))    # 最新的pattern。按列合并
            elif column_position == 4:
                cut_pattern_fourth = np.column_stack((cut_pattern_fourth, new_cut_pattern))    # 最新的pattern。按列合并
        stop += 1
        para_x = restricted_lp_master_problem(cut_pattern, cut_pattern_second, cut_pattern_third, cut_pattern_fourth, demand_number_array, roll_price)  # 将主问题松弛成线性问题
        flag_new_cut_pattern, new_cut_pattern, new_cut_pattern_price, column_position = knapsack_subproblem(para_x, demand_width_array, roll_width, roll_price)
    # 找到所有的cut pattern后，求解整数优化主问题。
    minimal_stock, optimal_number = restricted_ip_master_problem(cut_pattern, cut_pattern_second, cut_pattern_third, cut_pattern_fourth, demand_number_array, roll_price)
    end_time = time.time()
    # 生成结果，输出
    cp1, cp2, cp3, cp4 = np.transpose(cut_pattern).tolist(), np.transpose(cut_pattern_second).tolist(),np.transpose(cut_pattern_third).tolist(),np.transpose(cut_pattern_fourth).tolist()
    all_pattern = cp1 + cp2 + cp3 + cp4
    dict_stock = [0 for i in range(len(cp1))] + [1 for i in range(len(cp2))] + [2 for i in range(len(cp3))] + [3 for i in range(len(cp4))]
    import csv
    # 创建文件对象
    f = open('true_RESULT_FOUR_ROLL_'+str(stop_iteration)+".csv", 'w', encoding='utf-8', newline='' "")
    # 基于文件对象构建csv写入对象
    csv_writer = csv.writer(f)
    title = ["roll_width"] + ["roll_price"] + ["stock_len"+str(int(m)) for m in demand_width_array] + ["required"]
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
    csv_writer.writerow(["spend time", str(end_time - start_time)+"s"])
    f.close()
    #print(f'cut_pattern: {cut_pattern}')
    #print(f'cut_pattern_second: {cut_pattern_second}')
    #print(f'cut_pattern_third: {cut_pattern_third}')
    #print(f'cut_pattern_fourth: {cut_pattern_fourth}')
    #print(f'demand_width_array: {demand_width_array}')
    #print(f'demand_number_array: {demand_number_array}')
    print('result:')
    print(f'minimal_cost: {minimal_stock}')
    #print(f'optimal_number: {(optimal_number)}')
    print(f'require_number: {sum(optimal_number)}')
    print("time: ", end_time - start_time, "s")

def load_file(file_name):
    demand_width_array,demand_number_array = [],[]
    with open(file_name, encoding='utf-8') as file:
        file.readline()
        stop = False
        while stop==False:
            a = file.readline()
            print(a[0])
            if a[0]!='1':
                stop = True
                break
            a = a.strip('\n').split(',')
            width, num = int(a[2]), int(a[3])
            demand_width_array.append(width)
            demand_number_array.append(num)
    demand_width_array, demand_number_array = np.array(demand_width_array), np.array(demand_number_array)
    return demand_width_array, demand_number_array


def generate_data(roll_width, customer_num):
    roll_width = np.array(roll_width)
    roll_price = np.array([5, 9])
    np.random.seed(1)
    demand_width_array = np.random.randint(10, 100, size=(customer_num))
   # demand_width_array = demand_width_array/1000
    np.random.seed(2)
    demand_number_array = np.random.randint(10, 100, size=(customer_num))
    return roll_width, roll_price, demand_width_array, demand_number_array

def main(customer_count):
    """测试数据生成"""
    # roll_width, roll_price, demand_width_array, demand_number_array = generate_data(roll_width=[], customer_num=customer_count)
    roll_width = np.array([11000, 12000, 10000, 90000])
    roll_price = np.array([11, 12, 10, 90])
    #roll_width = np.array([11000])
    #roll_price = np.array([240])
    demand_width_array, demand_number_array = load_file("No5_factory.csv")
    """数据测试"""
    test(roll_width, roll_price, demand_width_array, demand_number_array, stop_iteration=customer_count)
    """检测输出的解是否合法"""
    path = "true_RESULT_FOUR_ROLL_"+str(customer_count)+".csv"
    result_validate(path, demand_width_array, demand_number_array)
    """保存 instance 文件"""
    instance = open("true_instance_"+str(customer_count)+".txt", "w")
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

customer_count_list = [100]
for i in customer_count_list:
    main(i)


