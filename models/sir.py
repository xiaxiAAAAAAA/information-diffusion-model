import scipy.integrate as spi
import numpy as np
import pylab as pl


t_start = 0.0

TS = 1.0      # 间隔（横坐标）
ND = 60.0     # 范围（横坐标）

t_end = ND
t_inc = TS
t_range = np.arange(t_start, t_end + t_inc, t_inc)
alpha1 = 0.38
beta1 = 0.13
INPUT_SIR = (0.8,0.2,0.0)

# 计算SIR中的微分方程Compute the differential equations in the SIR


def diff_eqs_sir(INP, t):
    Y = np.zeros((3))  # 大小为3的数组存储微分方程的结果
    V = INP
    Y[0] = - alpha1 * V[0] * V[1]  # S的变化率The rate of change in S
    Y[1] = alpha1 * V[0] * V[1] - beta1 * V[1]  # The rate of change in I
    Y[2] = beta1 * V[1]  # The rate of change in R
    return Y

RES2 = spi.odeint(diff_eqs_sir, INPUT_SIR, t_range)  # 求解微分方程

# print(RES2)

# Ploting
pl.rcParams['font.family'] = 'Times New Roman'
pl.figure(figsize=(8, 6))
pl.subplot(111)
pl.plot(RES2[:, 0], color='y', linestyle='-', marker='*', linewidth = '0.8',label='Susceptible')
pl.plot(RES2[:, 1], color='#d81e06', linestyle='-', marker='*', linewidth = '0.8',label='Infected')
pl.plot(RES2[:, 2], color='#008000', linestyle='-', marker='*', linewidth = '0.8',label='Recovery')

# 给特定点做标记

infection = RES2[:,1]
max = np.argmax(infection)  # 找到感染者数量最大值
max_index = infection[max]
# 添加一个注释，注释的内容为最大感染者数量对应的时间和数量
pl.annotate(r'['+str(max)+','+str(round(max_index,2))+']',  # 注释的文本内容，使用字符串拼接将时间和数量格式化为注释的形式
            xy=(max,infection[max]),  # 指定注释箭头所指向的坐标点
            xycoords='data',  # 表示注释箭头的坐标系是数据坐标系
            xytext=(+40,+10),   # 注释文本的偏移量，即相对于箭头指向的点的偏移量
            textcoords="offset points",   # 注释文本的坐标系是偏移坐标系
            fontsize = 18,
            color="#d81e06",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))   # 使用带有弧线的箭头

pl.legend(loc='center right', fontsize=18, bbox_to_anchor=(0.98, 0.7))  # 添加图例
pl.xlabel('Time', fontsize=18)
pl.ylabel('SIR Rate', fontsize=18)
pl.tick_params(labelsize=18)

pl.savefig("../plot/picture/sir_C_m.pdf",  dpi=1000, bbox_inches = 'tight')   # 自动调整图像周围的白边
pl.show()