# 複製gpt的
import scipy.optimize as opt

# 目標函數係數
c = [-3, -2, -5]  # 因為 linprog 是最小化問題，所以我們要把目標函數的係數取反

# 限制條件係數矩陣
A = [[1, 1, 0],
     [2, 0, 1],
     [0, 1, 2]]

# 限制條件的右側常數項
b = [10, 9, 11]

# 變數的非負限制
x_bounds = (0, None)
y_bounds = (0, None)
z_bounds = (0, None)

# 解線性規劃問題
result = opt.linprog(c, A_ub=A, b_ub=b, bounds=[x_bounds, y_bounds, z_bounds], method='highs')

# 輸出結果
if result.success:
    print(f"最優解: x = {result.x[0]:.2f}, y = {result.x[1]:.2f}, z = {result.x[2]:.2f}")
    print(f"最大值: {-result.fun:.2f}")  # 記得將結果取反
else:
    print("無法找到最優解")
