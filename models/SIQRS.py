import scipy.integrate as spi
import numpy as np
import matplotlib.pylab as pl

TS = 1.0      # 间隔（横坐标）import scipy.integrate as spi

alpha = 0.67
beta = 0.1
miu = (1-beta)*0.8

TS = 1.0      # 间隔（横坐标）
ND = 60.0     # 范围（横坐标）

S0 = 0.80
I0 = 0.1
Q0 = 0.1
R0 = 0.00

INPUT_STIR = (S0, I0, Q0, R0)

t_start = 0.0
t_end = ND
t_inc = TS
t_range = np.arange(t_start, t_end + t_inc, t_inc)

def diff_eqs(INP, t):
    Y = np.zeros((4))
    V = INP
    Y[0] = - alpha * V[0] * V[1]   # S
    Y[1] = alpha * V[0] * V[1] - beta * V[1] - miu * V[1] * V[2]  # I
    Y[2] = miu * V[1] * V[2] - beta * V[2]  # Q
    Y[3] = beta * V[1] + beta * V[2] # R
    return Y

RES1 = spi.odeint(diff_eqs, INPUT_STIR, t_range)

# print(RES1)

# Ploting
# 设置Matplotlib的全局字体为Times New Roman
pl.rcParams['font.family'] = 'Times New Roman'
pl.figure(figsize=(8, 6))

pl.subplot(111)

pl.plot(RES1[:, 0], color='y', linestyle='-', marker='*', linewidth = '0.8',label='Susceptible')
pl.plot(RES1[:, 1], color='#CF3D3E', linestyle='-', marker='*', linewidth = '0.8',label='Infect')
pl.plot(RES1[:, 2], color='#403990', linestyle='-', marker='*', linewidth = '0.8',label='Quarantine')
pl.plot(RES1[:, 3], color='#008000', linestyle='-', marker='*', linewidth = '0.8',label='Recovery')

## 给特定点做标记
infection = RES1[:, 1]
max = np.argmax(infection)  # 找到感染者数量最大值
max_index = infection[max]
pl.annotate(r'['+str(max)+','+str(round(max_index, 2))+']',  # 注释的文本内容，使用字符串拼接将时间和数量格式化为注释的形式
            xy=(max, infection[max]),  # 指定注释箭头所指向的坐标点
            xycoords='data',  # 表示注释箭头的坐标系是数据坐标系
            xytext=(+40, +10),  # 注释文本的偏移量，即相对于箭头指向的点的偏移量
            textcoords="offset points",  # 注释文本的坐标系是偏移坐标系
            fontsize=18,
            color="#d81e06",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))  # 使用带有弧线的箭头

pl.legend(loc='center right', fontsize=18, bbox_to_anchor=(0.98, 0.7))  # 添加图例
pl.xlabel('Time', fontsize=18)
pl.ylabel('SIQRS Rate', fontsize=18)
pl.tick_params(labelsize=18)

pl.savefig("../plot/picture/siqrs_C_m.pdf",  dpi=1000, bbox_inches='tight')  # 自动调整图像周围的白边
pl.show()
