import numpy as np
from sklearn.model_selection import train_test_split
from functionKNN import predict, accuracy_score

# Download dataset
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data # data
y = iris.target # label

# print(X)
# print(y)

# --- Random dữ liệu thủ công để chia data train, test ---

randomIndex = np.arange(X.shape[0]) # tạo list index từ 0 -> 149

np.random.shuffle(randomIndex) # Xáo trộn ngẫu nhiên theo kiểu cân bằng
# print(randomIndex)

X = X[randomIndex]
y = y[randomIndex]
# print(y)

# print(randomIndex)
# print(X)
# print(y)

# Cắt train, test
X_train = X[:100, :] # 100 train
X_test = X[100:, :] # 50 test
y_train = y[:100]
y_test = y[100:]

# --- Cắt bằng thư viện ---

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50)
# print(X_train)
# print(y_train)


# ----

y_predict = []
k = 5

for p in X_test:
	label = predict(X_train, y_train, p, k)
	y_predict.append(label)

# print(y_predict)
# print(y_test)

print("Accuracy: ", accuracy_score(y_predict, y_test))