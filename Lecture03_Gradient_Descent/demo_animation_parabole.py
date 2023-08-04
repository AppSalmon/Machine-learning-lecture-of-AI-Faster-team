import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
from sklearn import linear_model
import matplotlib.animation as animation

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
		g[i] = (cost(x1) - cost(x2))/(2*eps)	

	g_grad = grad(x)
	if np.linalg.norm(g-g_grad) > 1e-5:
		print("WARNING: CHECK GRADIENT FUNCTION!")

def gradient_descent(x_init, learning_rate, iteration):
	x_list = [x_init]
	m = A.shape[0]

	for i in range(iteration):
		x_new = x_list[-1] - learning_rate*grad(x_list[-1])
		if np.linalg.norm(grad(x_new))/m < 0.5: # when to stop GD
			break
		x_list.append(x_new)

	return x_list

# Data
b = np.array([[2,5,7,9,11,16,19,23,22,29,29,35,37,40,46,42,39,31,30,28,20,15,10,6]]).T
A = np.array([[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]]).T

# Draw data
fig1 = plt.figure("GD for Linear Regression")
ax = plt.axes(xlim=(-10,50), ylim=(-10,50)) 
plt.plot(A,b, 'ro')

# Add one to A
ones = np.ones((A.shape[0],1), dtype=np.int8)
A = np.concatenate((ones,A), axis=1)

# append x^2 to A
x_square = np.array([A[:,1]**2]).T
A = np.concatenate((A,x_square), axis=1)

# Draw parabole by formula
x_fomular = np.linalg.inv(A.transpose().dot(A)).dot(A.transpose()).dot(b)
x0_gd = np.linspace(1,46,10000)
y0_fomular = x_fomular[0][0] + x_fomular[1][0] * x0_gd + x_fomular[2][0] * x0_gd * x0_gd
plt.plot(x0_gd, y0_fomular, color='green')


# Random initial line
# x_init = np.array([[ 1.05455731],
#        [ 3],
#        [-0.01216413]])
# x_init = np.array([[ -2.1],
# 			       [ 3.1],
# 			       [-2.1]])
x_init = np.array([[ -2.1],
       [ 5.1],
       [-2.1]])
       
y0_init = x_init[0][0] + x_init[1][0]*x0_gd + x_init[2]*x0_gd*x0_gd
plt.plot(x0_gd,y0_init, color="black")

check_grad(x_init)

# Run gradient descent
iteration = 70
learning_rate = 0.000001

x_list = gradient_descent(x_init, learning_rate, iteration)

# plot black x_list
for i in range(len(x_list)):
	y0_x_list = x_list[i][0] + x_list[i][1]*x0_gd +  x_list[i][2]*x0_gd*x0_gd
	plt.plot(x0_gd, y0_x_list, color='black', alpha = 0.3)

# Draw animation
line , = ax.plot([],[], color = "blue")
def update(i):
	y0_gd = x_list[i][0][0] + x_list[i][1][0]*x0_gd + x_list[i][2][0]*x0_gd*x0_gd
	line.set_data(x0_gd, y0_gd)
	return line,

iters = np.arange(1,len(x_list), 1)
line_ani = animation.FuncAnimation(fig1, update, iters, interval=50, blit=True)

# legend for plot
plt.legend(('Value in each GD iteration', 'Solution by formular', 'Inital value for GD'), loc=(0.52, 0.01))
ltext = plt.gca().get_legend().get_texts()

# title
plt.title("Gradient Descent Animation")

plt.show()