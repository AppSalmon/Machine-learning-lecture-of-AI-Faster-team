import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data_classification.csv', header = None)

print(data.values)

true_x = []
true_y = []
false_x = []
false_y = []

for item in data.values:
	if item[2] == 1.:
		true_x.append(item[0])
		true_y.append(item[1])
	else:
		false_x.append(item[0])
		false_y.append(item[1])

# plt.scatter(true_x, true_y, marker = 'o', c = 'b')
# plt.scatter(false_x, false_y, marker = 'o', c = 'r')

# plt.show()

def sigmoid(z):
	return 1.0 / (1 + np.exp(-z))

def phanchia(p):
	if p >= 0.5:
		return 1
	return 0


def logistic_sigmoid_regression(X, y, w_init, eta, tol = 1e-4, max_count = 10000):
	w = [w_init]    
	it = 0
	N = X.shape[1]
	d = X.shape[0]
	count = 0
	check_w_after = 20
	while count < max_count:
		# mix data 
		mix_id = np.random.permutation(N)
		for i in mix_id:
			xi = X[:, i].reshape(d, 1)
			yi = y[i]
			zi = sigmoid(np.dot(w[-1].T, xi))
			w_new = w[-1] + eta*(yi - zi)*xi
			count += 1
			# stopping criteria
			if count%check_w_after == 0:                
				if np.linalg.norm(w_new - w[-check_w_after]) < tol:
					return w
			w.append(w_new)
	return w
   
def predict(features, weights):
	np.dot(feature, weights)
	return sigmoid(z)

def cost_function(features, labels, weights):
	n = len(labels)
	prediction == predict()