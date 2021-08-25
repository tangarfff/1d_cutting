"""Defines the main task for the VRP.

6/17更新：新增 stock规格，更改 reward
6.22 更新選擇鋼管的方式。

"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from random import choice


class VehicleRoutingDataset(Dataset):
    def __init__(self, num_samples, input_size, max_load=120, max_demand=60,
                 seed=None):
        super(VehicleRoutingDataset, self).__init__()
        stock_type = [60, 90, 96, 120]
        l = [1, 1.5, 1.6, 2]
        if max_load < max_demand:
            raise ValueError(':param max_load: must be > max_demand')

        if seed is None:
            seed = np.random.randint(1234567890)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.num_samples = num_samples
        self.max_load = max_load
        self.max_demand = max_demand

        dynamic_shape = (num_samples, 1, input_size + 1)
        # shape: [1000,1,21]
        loads = torch.full(dynamic_shape, choice(l))
        #loads = torch.array(dynamic_shape, choice(l))
        # All states will have their own intrinsic demand in [1, max_demand),  每一个state都有其特定demand，取值为[1,9]
        # then scaled by the maximum load. E.g. if load=10 and max_demand=30,
        # demands will be scaled to the range (0, 3) 转换demand
        #  shape: [1000,1,21]
        demands = torch.randint(1, max_demand + 1, dynamic_shape)
        # demands = demands / float(sum(stock_type))
        demands = demands / float(sum(stock_type))
        demands[:, 0, 0] = 0
        self.dynamic = torch.tensor(np.concatenate((loads, demands), axis=1))
        static_empty = torch.zeros(num_samples, 1, input_size + 1)
        d = demands.squeeze(1)  # 降维处理
        self.static = torch.stack((d, static_empty.squeeze(1)), 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # (static, dynamic, start_loc)
        b = self.static[idx, :, 0:1]
        return (self.static[idx], self.dynamic[idx], self.static[idx, :, 0:1])

    def update_mask(self, mask, dynamic, chosen_idx=None):
        """Updates the mask used to hide non-valid states.

        Parameters
        ----------
        dynamic: torch.autograd.Variable of size (1, num_feats, seq_len)
        """
        loads = dynamic.data[:, 0]  # 切割之后roll剩余的长度
        demands = dynamic.data[:, 1]  # 需要的钢管长度

        # 所有钢管都切割完，结束
        if demands.eq(0).all():
            return demands * 0.

        """demands=0代表钢管被切割完毕
        mask住demand=0的点以及demand>当前roll剩余长度的点"""
        new_mask = demands.ne(0) * demands.lt(loads)
        # print( demands.ne(0))
        # print( demands.lt(loads)) demand是否小于load
        # We should avoid traveling to the depot back-to-back
        repeat_home = chosen_idx.ne(0)
        if repeat_home.any():  # 或操作，任意一个元素为True，输出为True。
            new_mask[repeat_home.nonzero(), 0] = 1.  # 非0元素所在的行列
        if (~repeat_home).any():
            new_mask[(~repeat_home).nonzero(), 0] = 0.
        # ... unless we're waiting for all other samples in a mini-batch to finish
        has_no_load = loads[:, 0].eq(0).float()
        has_no_demand = demands[:, 1:].sum(1).eq(0).float()

        combined = (has_no_load + has_no_demand).gt(0)
        if combined.any():
            new_mask[combined.nonzero(), 0] = 1.
            new_mask[combined.nonzero(), 1:] = 0.

        return new_mask.float()

    def update_dynamic(self, dynamic, chosen_idx):
        """Updates the (load, demand) dataset values."""

        # Update the dynamic elements differently for if we visit depot vs. a city
        visit = chosen_idx.ne(0)
        depot = chosen_idx.eq(0)
        # print("depot",depot)
        # Clone the dynamic variable so we don't mess up graph
        all_loads = dynamic[:, 0].clone()
        all_demands = dynamic[:, 1].clone()

        load = torch.gather(all_loads, 1, chosen_idx.unsqueeze(1))
        demand = torch.gather(all_demands, 1, chosen_idx.unsqueeze(1))

        # Across the minibatch - if we've chosen to visit a city, try to satisfy
        # as much demand as possible
        if visit.any():

            new_load = torch.clamp(load - demand, min=0)
            new_demand = torch.clamp(demand - load, min=0)

            # Broadcast the load to all nodes, but update demand seperately
            visit_idx = visit.nonzero().squeeze()

            all_loads[visit_idx] = new_load[visit_idx]
            all_demands[visit_idx, chosen_idx[visit_idx]] = new_demand[visit_idx].view(-1)
            all_demands[visit_idx, 0] = -1. + new_load[visit_idx].view(-1)

        # Return to depot to fill vehicle load
        if depot.any():
            stock_type = [60, 90, 96, 120]
            l = [1, 1.5, 1.6, 2]
            """在这里增加多个木条选择，随机选一根木条"""
            all_loads[depot.nonzero().squeeze()] = choice(l)
            all_demands[depot.nonzero().squeeze(), 0] = 0.

        tensor = torch.cat((all_loads.unsqueeze(1), all_demands.unsqueeze(1)), 1)
        return torch.tensor(tensor.data, device=dynamic.device)

# TODO: 目标函数:1.浪费最少 2.MINIMUM STOCK
def reward(static, tour_indices):
    stock_num_list = []
    for i, _ in enumerate(tour_indices):
        stock_num = 0
        ax = tour_indices[i].cpu().numpy().flatten()
        where = np.where(ax == 0)[0]
        """这里需要更改：demand相加=多少，确定使用的哪一根钢管"""
        for j in range(len(where) - 1):
            low = where[j]
            high = where[j + 1]
            if low + 1 == high:
                continue
            stock_num += 1
        stock_num_list.append(stock_num)
    stock_num_reward = torch.tensor(stock_num_list).cuda().float()
    return stock_num_reward.float()


def render(static, tour_indices, save_path):
    """Plots the found solution."""

    plt.close('all')

    num_plots = 3 if int(np.sqrt(len(tour_indices))) >= 3 else 1

    _, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                           sharex='col', sharey='row')

    if num_plots == 1:
        axes = [[axes]]
    axes = [a for ax in axes for a in ax]

    for i, ax in enumerate(axes):

        # Convert the indices back into a tour
        idx = tour_indices[i]
        if len(idx.size()) == 1:
            idx = idx.unsqueeze(0)

        idx = idx.expand(static.size(1), -1)
        data = torch.gather(static[i].data, 1, idx).cpu().numpy()

        start = static[i, :, 0].cpu().data.numpy()
        x = np.hstack((start[0], data[0], start[0]))
        y = np.hstack((start[1], data[1], start[1]))

        # Assign each subtour a different colour & label in order traveled
        idx = np.hstack((0, tour_indices[i].cpu().numpy().flatten(), 0))
        where = np.where(idx == 0)[0]

        for j in range(len(where) - 1):

            low = where[j]
            high = where[j + 1]

            if low + 1 == high:
                continue

            ax.plot(x[low: high + 1], y[low: high + 1], zorder=1, label=j)

        ax.legend(loc="upper right", fontsize=3, framealpha=0.5)
        ax.scatter(x, y, s=4, c='r', zorder=2)
        ax.scatter(x[0], y[0], s=20, c='k', marker='*', zorder=3)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=200)


