import pygame
from random import randint
from math import sqrt
from sklearn.cluster import KMeans

def create_text(x, color): # Tạo chữ
	font = pygame.font.SysFont('sans', 40)
	return font.render(x, True, color)

def distance(p1, p2): # Tính khoảng cách 2 điểm
	return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)



pygame.init() # Khởi tạo pygame

screen = pygame.display.set_mode((1200, 700)) # Tạo màn hình

pygame.display.set_caption('AI Faster: Kmeans visualization') # Tên chương trình

icon = pygame.image.load('logo.jpg')
pygame.display.set_icon(icon) # Set logo

running = True
clock = pygame.time.Clock() # Tạo FPS

BACKGROUND = (214, 214, 214) # Màu background
BLACK = (0, 0, 0)
BACKGROUND_PANEL = (249, 255, 230) # Background để vẽ
WHITE = (255, 255, 255)

font_small = font = pygame.font.SysFont('sans', 15)

RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
YELLOW = (147, 153, 35)
PURPLE = (255,0,255)
SKY = (0,255,255)
ORANGE = (255,125,25)
GRAPE = (100,25,125)
GRASS = (55,155,65)

COLORS = [RED,GREEN,BLUE,YELLOW,PURPLE,SKY,ORANGE,GRAPE,GRASS]

# Data
K = 0
Error = 0
points = []
clusters_random = []
labels = []

while running:
	clock.tick(60)
	screen.fill(BACKGROUND)

	# ----- Draw interface ----- #

	mouse_x, mouse_y = pygame.mouse.get_pos() # Lấy tọa độ chuột
	
	# Draw panel
	pygame.draw.rect(screen, BLACK, (50, 50, 700, 500)) # rec: vẽ HCN
	pygame.draw.rect(screen, BACKGROUND_PANEL, (55, 55, 690, 490)) # rec: vẽ HCN

	# K button +
	pygame.draw.rect(screen, BLACK, (850, 50, 50, 50))
	screen.blit(create_text('+', WHITE), (865, 50))

	# K button -
	pygame.draw.rect(screen, BLACK, (950, 50, 50, 50))
	screen.blit(create_text('-', WHITE), (970, 50))

	# K value
	screen.blit(create_text('K = ' + str(K), BLACK), (1050, 50))
	
	# Run button
	pygame.draw.rect(screen, BLACK, (850, 150, 150, 50))
	screen.blit(create_text('Run', WHITE), (895, 150))

	# Random button
	pygame.draw.rect(screen, BLACK, (850, 250, 150, 50))
	screen.blit(create_text('Random', WHITE), (860, 250))

	# Algorithm use scikit-learn
	pygame.draw.rect(screen, BLACK, (850, 450, 150, 50))
	screen.blit(create_text('Algorithm', WHITE), (855, 450))

	# Reset buttom
	pygame.draw.rect(screen, BLACK, (850, 550, 150, 50))
	screen.blit(create_text('Reset', WHITE), (880, 550))

	

	# Draw mouse position when mouse is in panel
	if 50 < mouse_x < 750 and 50 < mouse_y < 550:
		screen.blit(font_small.render(f'({mouse_x-50}, {mouse_y-50})', True, BLACK), (mouse_x+15, mouse_y))



	# ----- End draw interface ----- #



	for event in pygame.event.get(): # Sự kiện

		# Đóng chương trình
		if event.type == pygame.QUIT:
			print('Quit program')
			running = False

		# Lúc bấm chuột
		if event.type == pygame.MOUSEBUTTONDOWN:

			# Create point on panel
			if 50 < mouse_x < 750 and 50 < mouse_y < 550:
				labels = []
				point = [mouse_x-50, mouse_y-50]
				points.append(point)


			# Tăng số cụm
			if 850 < mouse_x < 900 and 50 < mouse_y < 100: 
				print('Tăng số cụm')
				if K < 9:
					K += 1
				else:
					screen.blit(create_text('Erorr: K too much!', (190, 100, 100)), (50, 1))
			# Giảm số cụm	
			if 950 < mouse_x < 1000 and 50 < mouse_y < 100:
				print('Giảm số cụm')
				if K > 0:
					K -= 1
				else:
					screen.blit(create_text('Erorr: K < 0!', (190, 100, 100)), (50, 1))

			# Run
			if 800 < mouse_x < 1000 and 150 < mouse_y < 200:
				if clusters_random == []:
					screen.blit(create_text('Please create clusters...', (190, 100, 100)), (50, 1))
				else:

					# --- Kmeans thuần toán học --- #

					# Update label points
					labels = []
					for p in points:
						distances_to_cluster = []
						min_distance = 1e9
						for c in clusters_random:
							dis = distance(p, c)
							distances_to_cluster.append(dis)
							min_distance = min(min_distance, dis)
						label = distances_to_cluster.index(min_distance)
						labels.append(label)

					# Update cluster
					for i in range(K):
						sum_x = 0
						sum_y = 0
						count = 0
						for j in range(len(points)):
							if labels[j] == i:
								sum_x += points[j][0]
								sum_y += points[j][1]
								count += 1
						if count > 0:
							new_cluster_x = sum_x / count
							new_cluster_y = sum_y / count
							clusters_random[i] = [new_cluster_x, new_cluster_y]




			# Random button
			if 850 < mouse_x < 1000 and 250 < mouse_y < 300:
				clusters_random = []
				labels = []
				for i in range(K):
					clusters_random.append([randint(0, 700), randint(0, 500)])
				screen.blit(create_text('Random...', (190, 100, 100)), (50, 1))

			# Algorithm
			if 850 < mouse_x < 1000 and 450 < mouse_y < 500:
				if K == 0 or points == []:
					screen.blit(create_text('Error: K <= 0 or point is empty!', (190, 100, 100)), (50, 1))
					continue
					
				kmeans = KMeans(n_clusters = K).fit(points)
				clusters_random = kmeans.cluster_centers_
				labels = kmeans.predict(points)

			# Reset button
			if 850 < mouse_x < 1000 and 550 < mouse_y < 600:
				screen.blit(create_text('Reset...', (190, 100, 100)), (50, 1))
				K = 0
				Error = 0
				points = []
				clusters_random = []
				labels = []
	
	# Draw point
	for i in range(len(points)):
		points[i]
		pygame.draw.circle(screen, BLACK, (points[i][0]+50, points[i][1]+50), 5)
		if labels == []:
			pygame.draw.circle(screen, WHITE, (points[i][0]+50, points[i][1]+50), 4)
		else:
			pygame.draw.circle(screen, COLORS[labels[i]], (points[i][0]+50, points[i][1]+50), 4)

	# Draw clusters
	for i in range(len(clusters_random)):
		pygame.draw.circle(screen, COLORS[i], (int(clusters_random[i][0])+50, int(clusters_random[i][1])+50), 8)

	# Calculate error
	Error = 0
	if clusters_random != [] and labels != []:
		for i in range(len(points)):
			Error += distance(points[i], clusters_random[labels[i]])

	# Error
	screen.blit(create_text('Error: ' + str(int(Error)), BLACK), (850, 350))


	pygame.display.flip() # Show chương trình
pygame.quit()