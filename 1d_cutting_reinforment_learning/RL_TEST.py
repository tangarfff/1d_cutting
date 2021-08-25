import os
import numpy as np
import torch
from model import DRL4TSP
from tasks import vrp
import time
from tasks.vrp import VehicleRoutingDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_test_data(roll_width, demand_width_array, demand_number_array):
    max_load = roll_width
    max_demand = max(demand_number_array)
    input_size = sum(demand_number_array)
    dynamic_shape = (1, 1, input_size + 1)
    loads = torch.full(dynamic_shape, 1.)
    demands = torch.zeros(1)
    for i in range(len(demand_width_array)):
        single_demands = np.repeat(demand_width_array[i] / float(max_load),
                                   demand_number_array[i])  # torch.randint(1, max_demand + 1, dynamic_shape)
        single_demands = torch.from_numpy(single_demands).float()
        demands = torch.cat((demands, single_demands))
    demands = torch.reshape(demands, dynamic_shape)
    dynamic = torch.tensor(np.concatenate((loads, demands), axis=1))  # 动态输入的值
    static_empty = torch.zeros(1, 1, input_size + 1)
    d = demands.squeeze(1)  # 降维处理
    static = torch.stack((d, static_empty.squeeze(1)), 1)
    x = static[0, :, 0:1]
    x = torch.reshape(x, (1, 2, 1))
    return static, dynamic, x

def validate(actor_path, reward_fn, save_dir='.'):
    """Used to monitor progress on a validation set & optionally plot solution."""
    train_data = VehicleRoutingDataset(num_samples=1000,
                                       input_size=50,
                                       max_load=20,
                                       max_demand=9,
                                       seed=12345)
    actor = DRL4TSP(static_size=2,   # 2
                    dynamic_size=2,  # 2
                    hidden_size=128,  # 128
                    update_fn=train_data.update_dynamic,
                    mask_fn=train_data.update_mask,
                    num_layers=1,  # 1
                    dropout=0.1).to(device)
    path = os.path.join(actor_path, 'actor.pt')
    actor.load_state_dict(torch.load(path, device))
    actor.eval()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    t1 = time.time()
    rewards = []
    static, dynamic, x0 = generate_test_data(roll_width, demand_width_array, demand_number_array)
    static = static.to(device)
    dynamic = dynamic.to(device)
    x0 = x0.to(device) if len(x0) > 0 else None
    with torch.no_grad():
        tour_indices, _ = actor.forward(static, dynamic, x0)
    print("tour_indices: ",tour_indices)
    reward = reward_fn(static, tour_indices).mean().item()
    rewards.append(reward)
    t2 = time.time()
    print("reward: ", reward)
    print("time", t2-t1)
    actor.train()
    return np.mean(rewards), t2-t1

# roll_width, demand_width_array, demand_number_array = generate_data(roll_width=120, customer_num=5)
def load_file(file_name):
    demand_width_array,demand_number_array = [],[]
    with open(file_name, 'r') as file:
        n = int(file.readline().strip())
        capacity = int(file.readline().strip())
        for i in range(1, n + 1):
            a = file.readline().strip('\n').split('\t')
            # width, num = list(int(i) for i in file.readline().strip('\n').split('\t'))
            width, num = int(a[0]), int(a[-1])
            demand_width_array.append(width)
            demand_number_array.append(num)
    roll_width = np.array(capacity)
    demand_width_array, demand_number_array = np.array(demand_width_array), np.array(demand_number_array)
    return roll_width, demand_width_array, demand_number_array

# single test
roll_width, demand_width_array, demand_number_array = load_file("test_data/Falkenauer_u120_00.txt")
mean_valid = validate(actor_path="../trained_model", reward_fn=vrp.reward)
mean_valid, time_res = validate(actor_path="../trained_model", reward_fn=vrp.reward)

# bath test
"""source_path = "AI/"
fileList = os.listdir(source_path)
f = open("testResult_AI.csv", "w")
for i in fileList:
    StoreResult = []
    path = source_path+i
    roll_width, demand_width_array, demand_number_array = load_file(path)
    mean_valid, time_res = validate(actor_path="trained_model", reward_fn=vrp.reward)
    StoreResult.append(str(i))
    StoreResult.append(str(len(demand_number_array)))
    StoreResult.append(str(mean_valid))
    StoreResult.append(str(time_res))
    write_string = ' '.join(StoreResult)
    f.write(write_string + '\n')"""

