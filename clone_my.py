import csv
import cv2
import numpy as np

lines = []
with open('../data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
correction = 0.000001
for line in lines:
	for i in range(3):
		source_path = line[i]
		filename = source_path.split('/')[-1]
		current_path = '../data/IMG/' + filename
		image = cv2.imread(current_path)
		images.append(image)
	measurement = float(line[3])
	measurement_left = measurement + correction
	measurement_right = measurement - correction
	measurements.append(measurement)
	measurements.append(measurement_left)
	measurements.append(measurement_right)


X_train = np.array(images)
y_train = np.array(measurements)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D
from keras.layers.core import Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

X_train = np.array(X_train / 255.0 - 0.5)


#my model
model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(128, 3, 3, activation="relu"))
model.add(Convolution2D(128, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))
'''
#nvidia
model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
'''
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
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('../model_my17.h5')
