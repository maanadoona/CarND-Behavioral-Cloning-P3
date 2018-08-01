import csv
import cv2

lines = []
with open('../data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

for line in lines:
	for i in range(3):
		source_path = line[i]
		filename = source_path.split('/')[-1]
		current_path = '../data/IMG/' + filename
		image = cv2.imread(current_path)

		#crop
		height, width, channels = image.shape
		image_crop = image[45:height-25,0:width]

		#gray scale
		image_gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
		save_path = '../data/IMG_GRAY' + filename
		cv2.imwrite('../data/IMG_GRAY/' + filename, image_gray)
		
		#normailzation
		image_normal = cv2.normalize(image_gray, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		save_path = '../data/IMG_NORM/' + filename
		cv2.imwrite('../data/IMG_NORM/' + filename, image_normal)






