#Steering measurements range between -1 and 1
#The aim of CNN is to predict the steering measurements

import csv
import cv2
import numpy as np
import keras
import sklearn
import scipy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D
from keras.utils import np_utils
from sklearn.utils import shuffle
from keras.regularizers import l2, activity_l2
from sklearn.model_selection import train_test_split


#Open and load the driving_log.csv file
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)



#Training and Validation Split Sets
train_samples, validation_samples = train_test_split(lines, test_size=0.2, random_state=42)


#Image Preprocessing
#Apply Contrast_Limited_Adaptive_Histogram_Equalization to Y channel of all images of both train and test set
import cv2
def histogram_equal(image):
    """Apply Image histogram Equalization to Y Channel of Image 
    and convert the image from YUV to RGB"""
    yuv_channel = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y_luminance = yuv_channel[:,:,0]
    # create a CLAHE object 
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(3,3))
    y_ = clahe.apply(y_luminance)
    yuv_channel[:,:,0] = y_
    img_ = cv2.cvtColor(yuv_channel, cv2.COLOR_YUV2RGB)
    return img_




#Define the Python Data Generator 
def generator(samples, batch_size):
    """Python Image Generator Function"""
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                measurement = [float(batch_sample[3])]
                steering_left = measurement[0] + 0.2
                measurement.append(steering_left)
                steering_right = measurement[0] - 0.2
                measurement.append(steering_right)
                
                for i in range(3):
                    filename = batch_sample[i].split('\\')[-1]
                    current_path = './data/IMG/'+ filename
                    image = cv2.imread(current_path)
                    image = histogram_equal(image)
                    #image = cv2.resize(image, (200,112))
                    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    #plt.imshow(image)
                    images.append(image)
                    measurements.append(measurement[i])
                    #print(len(images), len(measurements))

                augmented_images = []
                augmented_measurements = []
                
                for image, measurement in zip(images, measurements):
                    augmented_images.append(image)
                    augmented_measurements.append(measurement)
                    image_flipped = cv2.flip(image, 1)
                    measurement_flipped = measurement*(-1.0)
                    augmented_images.append(image_flipped)
                    augmented_measurements.append(measurement_flipped)

                    # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            #print(X_train.shape, y_train.shape)
            yield sklearn.utils.shuffle(X_train, y_train)

#Assign the python generator
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)



#Modified Nvidia CNN Architecture, Trainable params: 981,819
model =  Sequential()

#Image Preprocessing
#Crop the image to get rid of horizon and car hood
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))

#Image Preprocessing
#Normalize the image
model.add(Lambda(lambda x:(x/255.0)-0.5))

#Add 3 Convolutional layers with filter size =5x5 and Stride=2x2  and  Relu Activation
#Layer 1
model.add(Convolution2D(24, 5,5, subsample=(2, 2), border_mode='valid'))
model.add(Activation('relu'))

#Layer 2
model.add(Convolution2D(36,5,5, subsample=(2, 2), W_regularizer=l2(0.001)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

#Layer 3
model.add(Convolution2D(48,5,5, subsample=(2, 2), W_regularizer=l2(0.001)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

#Add 3 Convolutional layers with filter size =3x3 and Stride=None  and  Relu Activation
#Layer 4
model.add(Convolution2D(64,3,3, W_regularizer=l2(0.001)))
model.add(Activation('relu'))
#Layer 5
model.add(Convolution2D(64,3,3, W_regularizer=l2(0.001)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

#Flatten the layer to output = 1152 neurons
model.add(Flatten())

#Layer 6: Fully Connected 1
model.add(Dense(100, W_regularizer=l2(0.001)))
model.add(Activation('relu'))
#Add Dropout Regularization
model.add(Dropout(0.2))

#Layer 7: Fully Connected 2
model.add(Dense(50, W_regularizer=l2(0.001)))
model.add(Activation('relu'))
#Add Dropout Regularization
model.add(Dropout(0.2))

#Layer 8: Fully Connected 3
model.add(Dense(10, W_regularizer=l2(0.001)))
model.add(Activation('relu'))
#Add Dropout Regularization
model.add(Dropout(0.5))

#Layer 9: Logit
model.add(Dense(1))


#Train the model
model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split = 0.2, shuffle=True, nb_epoch=7, batch_size=512, verbose = 1)
history = model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=10, verbose=1)

#print the model summary
print(model.summary())

#Save the model
model.save('model.h5')

