import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy

img = plt.imread("bird.jpg")

width = img.shape[0]
height = img.shape[1]

# print(img.shape)
# print(img)

img = img.reshape(width * height, 3)

# print(img.shape)
# print(img)

kmeans = KMeans(n_clusters=5).fit(img)

labels = kmeans.predict(img)
clusters = kmeans.cluster_centers_

print(labels)
print(clusters)

img2 = numpy.zeros_like(img)

print(img2)

for i in range(len(img2)):
	img2[i] = clusters[labels[i]]

img2 = img2.reshape(width,height,3)

plt.imshow(img2)
plt.show()
