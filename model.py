import csv
import cv2
import numpy as np
import sklearn
from random import shuffle

samples = []
with open('../data/driving_log.csv', encoding="utf-8") as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

correction = 0.1

def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1:
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			images = []
			angles = []
			for batch_sample in batch_samples:
				for i in range(3):
					name = '../data/IMG/' + batch_sample[i].split('/')[-1]
					image = cv2.imread(name)
					images.append(image)
					if i == 0:
						images.append(cv2.flip(image, 1))
	
				angle = float(batch_sample[3])
				angle_left = angle + correction
				angle_right = angle - correction				
				angles.append(angle)
				angles.append(angle*-1.0)
				angles.append(angle_left)
				angles.append(angle_right)

			X_train = np.array(images)
			X_train = np.array(X_train / 127.5 - 1.)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D, Dropout
from keras.layers.core import Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

#my model
model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Convolution2D(16, 5, 5, subsample=(2, 2), border_mode='valid', activation="relu"))
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode='valid', activation="relu"))
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode='valid', activation="relu"))
model.add(Convolution2D(128, 3, 3, activation="relu"))
model.add(Convolution2D(128, 3, 3, activation="relu"))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
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

model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*4, validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('../model.h5')


