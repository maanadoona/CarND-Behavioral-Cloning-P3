import csv
import cv2
import numpy as np

lines = []
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
for line in lines:
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = './data/IMG/' + filename
	image = cv2.imread(current_path)
	images.append(image)
	measurement = float(line[3])
	measurements.append(measurement)


X_train = np.array(images)
y_train = np.array(measurements)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D
from keras.layers.core import Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
X_normalized = np.array(X_train / 255.0 - 0.5)
#model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
#model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu", input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(1))
'''
# lenet
model = Sequential()
X_normalized = np.array(X_train / 255.0 - 0.5 )
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320, 3)))
model.add(Convolution2D(6,5,5,activation="relu", input_shape=(160, 320, 3)))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
'''
model.compile(loss='mse', optimizer='adam')
model.fit(X_normalized, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')
