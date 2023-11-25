import numpy as np
import matplotlib.pyplot as plt
alpha=0.00001
def f(p_t):
    #return -(np.abs(p_t-alpha))*np.log(np.abs(p_t-alpha))-(np.abs((1-p_t)-alpha))*np.log(np.abs((1-p_t)-alpha))
    return -(np.abs(p_t-0.5))*np.log(np.abs(p_t-0.5))*(1-p_t)**0.75
# 生成0到1之间的一系列p_t值
p_t_values = np.linspace(0, 1, 10000000)

# 计算对应的f(p_t)值
f_values = f(p_t_values)

# 绘制图像
plt.plot(p_t_values, f_values)
plt.xlabel('p_t')
plt.ylabel('f(p_t)')
plt.title('Graph of f(p_t)')
plt.grid(True)
plt.show()