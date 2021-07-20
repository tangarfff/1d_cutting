# 1d_cutting
## 分支定价法 (python) 只支持小规模数据集
dataset: 标准数据集

## 列生成算法 (python)
res_11m_300s: 原材料11m, 300s的运行结果。

res_12m_300s: 原材料12m, 300s的运行结果。

test_result： 不同规模原材料, 7200s的运行结果。 

column_generation.py 支持一种原材料的列生成。

gurobi_cg_four.py 支持四种原材料的列生成。

result_validation.py  列生成结果验证

## 降序首次适应算法(FFD)&降序最佳适应算法(BFD) (C#)
### 降序首次适应算法(FFD)
先对物品按降序排序，再按照首次适应算法进行装箱。

### 降序最佳适应算法(BFD)
先对物品按降序排序，再按照最佳适应算法进行装箱。
program.cs 主代码，可测试5号杠杆厂编号为1-7的真实数据

## TABU禁忌算法(python)&SA模拟退火算法(JAVA)
tabu参考 : https://blog.csdn.net/weixin_41565013/article/details/117927722

SA 参考：https://github.com/jakubBienczyk/1D-Cutting-Stock-Problem-Simulated-Annealing

## 测试结果
cutting_stock_result：测试结果
