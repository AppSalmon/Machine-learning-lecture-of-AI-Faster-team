import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn import linear_model
A = [2, 5, 7, 9, 11, 16, 19, 23, 22, 29, 29, 35, 37, 40, 46]
b = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
A = np.array([A]).T # Doi thanh hang doc
b = np.array([b]).T

LR = linear_model.LinearRegression()

#Train
LR.fit(A, b)
print(LR.coef_) # a
print(LR.intercept_) # b (độ dốc)

plt.plot(A, b, 'ro')

#Predict
x_test = 12
y_test = x_test*LR.coef_ + LR.intercept_
print(f"Predict for {x_test} là: {y_test}")

#Estimate (chuẩn đoán 1 đống dữ liệu)
x0 = np.array([[1, 46]]).T
y0 = x0*LR.coef_ + LR.intercept_

plt.plot(x0, y0)
plt.show()


