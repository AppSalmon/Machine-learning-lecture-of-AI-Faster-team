import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#random data
A = [2, 5, 7, 9, 11, 16, 19, 23, 22, 29, 29, 35, 37, 40, 46]
b = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# x = (A^T A)^-1 A^T y
A = np.array([A]).T # Doi thanh hang doc
b = np.array([b]).T
plt.plot(A, b, 'ro')
ones = np.ones((A.shape[0], A.shape[1]), dtype = np.int8) # Tao vector 1
A = np.concatenate((A, ones), axis = 1) # Gop vector them vao ben phai
x = np.linalg.inv(A.transpose().dot(A)).dot(A.transpose()).dot(b)
print(x)

#Du doan
x_test = 12
y_test = x_test*x[0][0] + x[1][0]
print(f"Predict for {x_test} là: {y_test}")

#Visualize data
x0 = np.array([[1, 46]]).T
y0 = x0*x[0][0] + x[1][0]
plt.plot(x0, y0)
plt.show()

