from math import sqrt
import operator

def cal_distances(p1, p2):
	# a -> b = sqrt((xa - xb)**2 + (ya - yb)**2 + (za - zb)**2 )
	dimension = len(p1) # Số chiều
	distance = 0
	for i in range(dimension):
		distance += (p1[i] - p2[i]) * (p1[i] - p2[i])

	return sqrt(distance)


def get_k_neighbors(train_X, label_y, point, k): # Lấy k point gần nhất
	distances = [] # Danh sach chua kc cac diem den point
	neighbors_labels = [] 
	
	for i in range(len(train_X)):
		distance = cal_distances(train_X[i], point)
		distances.append((distance, label_y[i])) 

	distances.sort(key = operator.itemgetter(0))

	for i in range(k):
		neighbors_labels.append(distances[i][1])

	return neighbors_labels


def highest_votes(labels): # Xem label phổ biến nhất và return
	labels_count = [0, 0, 0]

	for label in labels:
		labels_count[label] += 1

	return labels_count.index(max(labels_count))

def predict(train_X, label_y, point, k):
	neighbors_labels = get_k_neighbors(train_X, label_y, point, k)
	return highest_votes(neighbors_labels)

def accuracy_score(predict, labels):
	count = 0
	for i in range(len(predict)):
		if predict[i] == labels[i]:
			count += 1

	accuracy = count / len(predict)
	return accuracy
