"""验证切割结果"""
import pandas as pd
import numpy as np

def result_validate(path, demand_width_array, demand_number_array):
    data = pd.read_csv(path)
    labels = list(data.columns.values)
    # 检测每一个stock生产的数量够不够
    for i in range(len(demand_width_array)):
        m = data[labels[i + 2]] * data[labels[-1]]  # 记录每一列的数据
        c = sum(i for i in m if i > 0)
        if c < demand_number_array[i]:
            print("invalid result: ", demand_width_array[i], " need ", demand_number_array[i], " the result is ", c)
            # return False
    # 检测每一个切割方式是否合法
    cc = 0
    pattern_count = data.index.stop - 3   # 切割方式总数
    wasted_width_list = []
    true_used_width_list = []  # 每一个pattern的废料长度
    for i in range(pattern_count):
        single_pattern = data.iloc[i]
        rw = single_pattern[0]
        stock_list = np.array(list(map(int, single_pattern[2:-1])))
        used_width = sum(stock_list*np.array(demand_width_array))
        true_used_width_list.append(used_width)
        wasted_width_list.append(int(rw)-used_width)
        if int(rw) < used_width-0.001:
            print("invalid result: cutting pattern exceed the roll width")
            #return False
    # 检测总价格是否正确
    pattern_length_list = data[labels[0]][0:pattern_count]
    pattern_price_list = data[labels[1]][0:pattern_count]
    pattern_count_list = data[labels[-1]][0:pattern_count]
    pattern_length_list = np.array(list(map(int, pattern_length_list)))
    pattern_price_list = np.array(list(map(int, pattern_price_list)))
    pattern_count_list = np.array(list(map(int, pattern_count_list)))
    minimal_cost = sum(pattern_count_list * pattern_price_list)
    require_number = sum(pattern_count_list)
    total_wasted = wasted_width_list * pattern_count_list
    # print(sum(total_wasted))
    # print(sum(pattern_count_list))
    # print(sum(true_used_width_list*pattern_count_list))  # demand_width_array* demand_number_array
    total_used = sum(pattern_length_list * pattern_count_list)
    # print("minimal_cost ", minimal_cost)
    print("require_number ", require_number)
    print("total used length ", total_used)
    print("t d", sum(demand_width_array * demand_number_array))
    print("total wasted length ", total_wasted)
    print("废料占比 ", ((total_used-sum(demand_width_array * demand_number_array)))/total_used)
    print("valid result! ")
    print("wasted_width_list", wasted_width_list)
    return True








