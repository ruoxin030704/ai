# 參考網路資料，問chatgpt
import numpy as np  
from numpy.linalg import norm  
from micrograd.engine import Value  

# 使用梯度下降法尋找函數最低點
def gradientDescendent(f, p0, h=0.01, max_loops=100000, dump_period=1000):
    p = p0.copy()  # 初始化參數 p 為 p0 的副本
    for i in range(max_loops):  
        fp = f(p)  
        gp, t = [], []  # 初始化梯度列表 gp 和 Value 對象列表 t
        for j in range(len(p)):  
            t.append(Value(p[j])) 
        f(t).backward()  # 計算梯度
        for j in t:  
            gp.append(j.grad)  
        glen = norm(gp)  
        if i % dump_period == 0: 
            print('{:05d}: f(p) = {:.3f}, p = {}, gp = {}, glen = {:.5f}'.format(i, fp, str(p), str(gp), glen))
        if glen < 0.00001:  
            break
        gh = np.multiply(gp, -1 * h)  
        p += gh 
    print('{:05d}: f(p) = {:.3f}, p = {}, gp = {}, glen = {:.5f}'.format(i, fp, str(p), str(gp), glen))
    return p  

# 定義目標函數
def f(p):
    [x, y, z] = p  # 拆解參數列表
    return (x - 1) ** 2 + (y - 2) ** 2 + (z - 3) ** 2  

# 初始參數
p = [0.0, 0.0, 0.0]  

# 執行梯度下降法
gradientDescendent(f, p)  # 尋找函數的最低點
