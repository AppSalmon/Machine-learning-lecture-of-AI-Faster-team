# Code ở đây cũng được
import numpy as np
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
from sklearn import linear_model
import random
import matplotlib.animation as animation
df = pd.read_csv("Data_female.txt", sep="\t")
df['Chiều cao'] = df['Chiều cao'].apply(lambda x: float(x.replace('cm', '')))
df['Cân nặng'] = df['Cân nặng'].apply(lambda x: float(x.replace('kg', '')))


def cost(x):
    m = A.shape[0]
    return 0.5/m * np.linalg.norm(A.dot(x) - b, 2)**2

def grad(x):
    m = A.shape[0]
    return 1/m * A.T.dot(A.dot(x)-b)

def check_grad(x):
    eps = 1e-4
    g = np.zeros_like(x)
    for i in range(len(x)):
        x1 = x.copy()
        x2 = x.copy()
        x1[i] += eps
        x2[i] -= eps
        g[i] = (cost(x1) - cost(x2)) / (2*eps) # Cong thuc f'(x) = (f(x + eps) - f(x - eps)) / 2eps
    g_grad = grad(x)
    if np.linalg.norm(g - g_grad) > 1e-7:
        print("Warning: Check gradient function!")



def gradient_descent(x_init, learning_rate, iteration):
    x_list = [x_init]
    m = A.shape[0]

    for i in range(iteration):
        x_new = x_list[-1] - learning_rate*grad(x_list[-1])

        if np.linalg.norm(grad(x_new))/ len(x_new) < 0.5: # when to stop GD
            break
        x_list.append(x_new)

    return x_list


# Data


A = np.array([df['Chiều cao']]).T
b = np.array([df['Cân nặng']]).T

# Visual data
fig1 = plt.figure("Gradient Descent for Linear Regression")
ax = plt.axes(xlim = (70, 200), ylim = (0, 70))
plt.plot(A, b, 'ro')

# Visual line Linear Regression
LR = linear_model.LinearRegression()
LR.fit(A, b)

x0_gd = np.linspace(70, 200, 2)
y0_sklearn = LR.intercept_[0] + LR.coef_[0][0] * x0_gd

plt.plot(x0_gd, y0_sklearn, color = 'green')
print(LR.intercept_, LR.coef_)

# Add one to A
ones = np.ones((A.shape[0],1), dtype=np.int8)
A = np.concatenate((ones,A), axis=1)

# Random initial line
x_init = np.array([[float(LR.intercept_[0]) + 5], [float(LR.coef_[0][0]) + 0.4]])
y0_init = x_init[0][0] + x_init[1][0]*x0_gd #b = ax
plt.plot(x0_gd, y0_init, color = "black")


# check_grad(x_init)

# Run gradient descent
iteration = 1000
learning_rate = 0.000005

x_list = gradient_descent(x_init, learning_rate, iteration)

# plot black x_list
for i in range(len(x_list)):
    y0_x_list = x_list[i][0] + x_list[i][1]*x0_gd
    plt.plot(x0_gd, y0_x_list, color='black', alpha = 0.3)

# Draw animation
line , = ax.plot([],[], color = "blue")

def update(i):
    y0_gd = x_list[i][0][0] + x_list[i][1][0]*x0_gd
    line.set_data(x0_gd, y0_gd)
    return line,

iters = np.arange(1,len(x_list), 1)
line_ani = animation.FuncAnimation(fig1, update, iters, interval=50, blit=True)

iters = np.arange(1,len(x_list), 1)
line_ani = animation.FuncAnimation(fig1, update, iters, interval=50, blit=True)

# legend for plot
plt.legend(('Điểm dữ liệu', 'Vector kết quả', 'Vector random khởi tạo', 'Mỗi lần chạy GD'), loc=(0.52, 0.01))

# title
plt.title("Gradient Descent Animation")

plt.show() #------------------------------------------------
# Ve su lien he giua cost va interation de tim diem dung tot

cost_list = []
iter_list = []
for i in range(len(x_list)):
    iter_list.append(i)
    cost_list.append(cost(x_list[i]))

plt.plot(iter_list, cost_list)
plt.xlabel('Interation')
plt.ylabel('Cost')
plt.show() #------------------------------------------------