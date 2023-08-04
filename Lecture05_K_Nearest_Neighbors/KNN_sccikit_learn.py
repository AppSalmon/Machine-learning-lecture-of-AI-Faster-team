from sklearn import datasets, neighbors
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
iris_X = iris.data # data 
iris_y = iris.target # label

randIndex = np.arange(iris_X.shape[0])
np.random.shuffle(randIndex)

iris_X = iris_X[randIndex]
iris_y = iris_y[randIndex]

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=50)

knn = neighbors.KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)
y_predict = knn.predict(X_test)

accuracy = accuracy_score(y_predict, y_test)
print(accuracy)