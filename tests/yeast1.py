
# 打开数据集文件
with open('./imbalance_dataset/yeast/yeast1.dat', 'r') as file:
    # 逐行读取数据
    data = file.readlines()

# 初始化X和y列表
X = []
y = []

# 遍历每一行数据
for line in data:
    # 去除换行符并按逗号分割数据
    line = line.strip().split(',')

    # 提取X，将每个特征转换为浮点数并添加到X列表
    X.append([float(x) for x in line[:-1]])

    # 提取y，将标签添加到y列表
    if line[-1] == ' negative':
        y.append(0)
    else:
        y.append(1)

# 打印X和y的示例数据
print("X:", X[:10])
print("y:", y[:10])