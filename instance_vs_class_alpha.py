import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 设置常数和自变量的范围
alpha_values = [0.25, 0.5, 1, 2, 4]
class_dependent_cost = 1
p_t_values = np.linspace(0, 1, 1000)

# 计算instance-dependent-cost
instance_dependent_cost = []
for alpha in alpha_values:
    cost = (1 + (1 - p_t_values) ** alpha) * class_dependent_cost
    instance_dependent_cost.append(cost)

# 绘制图像
sns.set(style="darkgrid")
plt.figure(figsize=(10, 6))
for i, alpha in enumerate(alpha_values):
    plt.plot(p_t_values, instance_dependent_cost[i], label=f"$\\alpha$={alpha}")

plt.xlabel("probability of ground truth class")
plt.ylabel("instance-dependent cost")
plt.legend()

# 标注公式
plt.text(0.5, 1.2, r"$instance-dependent \ cost = (1 + (1-p_t)^{\alpha}) \times class-dependent \ cost$",
         fontsize=12, ha="center")

# 添加标题
#plt.title("instance-dependent cost with different value of $\\alpha$")

plt.show()