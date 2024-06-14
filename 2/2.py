import random

# 定義城市座標
citys = [
    (0, 3), (0, 0),
    (0, 2), (0, 1),
    (1, 0), (1, 3),
    (2, 0), (2, 3),
    (3, 0), (3, 3),
    (3, 1), (3, 2)
]

# 初始化路徑，從0到城市數量-1
l = len(citys)
path = [i for i in range(l)]
print("初始路徑:", path)

# 計算兩個城市之間的距離
def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5

# 計算給定路徑的總距離
def pathLength(p):
    dist = 0
    plen = len(p)
    for i in range(plen):
        dist += distance(citys[p[i]], citys[p[(i + 1) % plen]])
    return dist

# 打印初始路徑長度
print("初始路徑長度:", pathLength(path))

# 爬山演算法來優化路徑
def hill_climb(path):
    current_length = pathLength(path)
    for _ in range(10000):  # 設置迭代次數
        # 隨機選擇兩個位置進行交換
        i, j = random.sample(range(len(path)), 2)
        new_path = path[:]
        new_path[i], new_path[j] = new_path[j], new_path[i]
        new_length = pathLength(new_path)
        # 如果新路徑更短，則接受新路徑
        if new_length < current_length:
            path = new_path
            current_length = new_length
    return path

# 使用爬山演算法優化路徑
optimized_path = hill_climb(path)
print("優化後的路徑:", optimized_path)
print("優化後的路徑長度:", pathLength(optimized_path))
