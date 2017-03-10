# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/P3_Architecture.png "CNN Architecture Visualization"
[image2]: ./examples/Udacity_Dataset_Visualization.JPG "Udacity Provided Dataset with Left and Right Image Augmentation"
[image3]: ./examples/Augmented_Dataset_Visualization.JPG "Dataset with new Vehicle Recovery from Shoulders"


####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 containing the video of autonomous model recording
* README.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the Python Generator pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. I have Used the Nvidia Architecture suggested in the Course and Modified it. The CNN Architecture is as follows:

##### Modified Nvidia CNN Architecture:  Loss Function: Mean Squared Error, Optimizer: Adam, Trainable Params: 981,819
* Layer 1:  Input Shape: 90x320x3, Filter: 5x5, Stride: 2x2, Output: 43x 158x24 , Activation: Relu
* Layer 2:  Input Shape: 43x 158x 24 , Filter: 5x5, Stride: 2x2, Output: 20,x77x36, Regularization: L2(0.001), Activation: Relu, Dropout: 0.5
* Layer 3:  Input Shape: 20,x77x36,, Filter: 5x5, Stride: 2x2, Output: 8x37x48, Regularization: L2(0.001), Activation: Relu, Dropout: 0.5
* Layer 4:  Input Shape: 8x37x48, Filter: 3x3, Stride: None, Output: 6x35x64, Regularization: L2(0.001), Activation: Relu
* Layer 5:  Input Shape: 6x35x64, Filter: 3x3, Stride: None, Output: 4x33x64, Regularization: L2(0.001), Activation: Relu, Dropout: 0.5
* Flatten: Output: 4x33x64 = 8448
* Fully Connected Layer 6:  Input Shape: 8448,  Output: 100, Regularization: L2(0.001), Activation: Relu, Dropout: 0.2
* Fully Connected Layer 7:  Input Shape: 100, Output: 50, Regularization: L2(0.001), Activation: Relu, Dropout: 0.2
* Fully Connected Layer 8:  Input Shape: 50, Output: 10, Regularization: L2(0.001), Activation: Relu, Dropout: 0.5
* Layer 9:  LOGIT = 1 Neuron – Steering Measurement Prediction
![alt text][image1]

####2. Attempts to reduce overfitting in the model

The Model has been heavily regularized using Dropout and L2 regularization. Since the total parameters that are trained is 981,819, and dataset I have used is highly skewed to center steering there are higher chances of overfitting. 
This model works fine in the simulator. Given the interest of time and limitation on the VARIETY of data used, I am sure this can be further improved even with less trainable parameters. This model is not expected to work well on the track 2 and for the reasons that model is not enough generalized under different driving conditions. This model has cloned my behavior to drive on track one. 

####3. Model parameter tuning

* The model used an Adam optimizer, so the learning rate was not tuned manually (model.py line 164).
* Epoch Used = 10
* Loss Function: MSE
* Dropouts used throughout and L2 Regularization Used

####4. Appropriate training data
##### Data Preprocessing
Image Preprocessing:
-	Apply Contrast Limited Adaptive Histogram Equalization to Y channel. (Line 35 model.py)
-	Crop the image to remove the irrelevant information like horizon and car hood (Line 108 model.py)
-	Normalize the image for faster convergence. (Line 112 model.py)
Python Image Generators for Faster Running of Model. (Line 51 model.py)
##### Data Augmentation
Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, Left Camera and Right Camera augmented images and recovering from the left and right sides of the road. The Udacity provided was heavily skewed to center driving. So I have also tried to incorporate driving data for recovery from road shoulders by recording videos from simulator.

##### Udacity Provided Dataset with Left Right Augmentation:
![alt text][image2]

##### Dataset with new Vehicle Recovery from Shoulders
![alt text][image3]

For details about how I created the training data, see the next section.
####5. Autonomous Driving Video recording
I have also included the video.mp4 file.

####7. Additional Info:  Training Summary
Epoch 1/10
8832/8666 [==============================] - 33s - loss: 0.3972 - val_loss: 0.2517
Epoch 2/10
8832/8666 [==============================] - 27s - loss: 0.2089 - val_loss: 0.1629
Epoch 3/10
8832/8666 [==============================] - 27s - loss: 0.1443 - val_loss: 0.1373
Epoch 4/10
8832/8666 [==============================] - 27s - loss: 0.1275 - val_loss: 0.1130
Epoch 5/10
8832/8666 [==============================] - 27s - loss: 0.1077 - val_loss: 0.0928
Epoch 6/10
8796/8666 [==============================] - 27s - loss: 0.0976 - val_loss: 0.0868
Epoch 7/10
8832/8666 [==============================] - 27s - loss: 0.0962 - val_loss: 0.1003
Epoch 8/10
8832/8666 [==============================] - 27s - loss: 0.0998 - val_loss: 0.0855
Epoch 9/10
8832/8666 [==============================] - 27s - loss: 0.0901 - val_loss: 0.0872
Epoch 10/10
8832/8666 [==============================] - 27s - loss: 0.0958 - val_loss: 0.0798
  ____________________________________________________________________________________________________
  Layer (type)                     Output Shape          Param #     Connected to
  ====================================================================================================
  cropping2d_1 (Cropping2D)        (None, 90, 320, 3)    0           cropping2d_input_1[0][0]
  ____________________________________________________________________________________________________
  lambda_1 (Lambda)                (None, 90, 320, 3)    0           cropping2d_1[0][0]
  ____________________________________________________________________________________________________
  convolution2d_1 (Convolution2D)  (None, 43, 158, 24)   1824        lambda_1[0][0]
  ____________________________________________________________________________________________________
  activation_1 (Activation)        (None, 43, 158, 24)   0           convolution2d_1[0][0]
  ____________________________________________________________________________________________________
  convolution2d_2 (Convolution2D)  (None, 20, 77, 36)    21636       activation_1[0][0]
  ____________________________________________________________________________________________________
  activation_2 (Activation)        (None, 20, 77, 36)    0           convolution2d_2[0][0]
  ____________________________________________________________________________________________________
  dropout_1 (Dropout)              (None, 20, 77, 36)    0           activation_2[0][0]
  ____________________________________________________________________________________________________
  convolution2d_3 (Convolution2D)  (None, 8, 37, 48)     43248       dropout_1[0][0]
  ____________________________________________________________________________________________________
  activation_3 (Activation)        (None, 8, 37, 48)     0           convolution2d_3[0][0]
  ____________________________________________________________________________________________________
  dropout_2 (Dropout)              (None, 8, 37, 48)     0           activation_3[0][0]
  ____________________________________________________________________________________________________
  convolution2d_4 (Convolution2D)  (None, 6, 35, 64)     27712       dropout_2[0][0]
  ____________________________________________________________________________________________________
  activation_4 (Activation)        (None, 6, 35, 64)     0           convolution2d_4[0][0]
  ____________________________________________________________________________________________________
  convolution2d_5 (Convolution2D)  (None, 4, 33, 64)     36928       activation_4[0][0]
  ____________________________________________________________________________________________________
  activation_5 (Activation)        (None, 4, 33, 64)     0           convolution2d_5[0][0]
  ____________________________________________________________________________________________________
  dropout_3 (Dropout)              (None, 4, 33, 64)     0           activation_5[0][0]
  ____________________________________________________________________________________________________
  flatten_1 (Flatten)              (None, 8448)          0           dropout_3[0][0]
  ____________________________________________________________________________________________________
  dense_1 (Dense)                  (None, 100)           844900      flatten_1[0][0]
  ____________________________________________________________________________________________________
  activation_6 (Activation)        (None, 100)           0           dense_1[0][0]
  ____________________________________________________________________________________________________
  dropout_4 (Dropout)              (None, 100)           0           activation_6[0][0]
  ____________________________________________________________________________________________________
  dense_2 (Dense)                  (None, 50)            5050        dropout_4[0][0]
  ____________________________________________________________________________________________________
  activation_7 (Activation)        (None, 50)            0           dense_2[0][0]
  ____________________________________________________________________________________________________
  dropout_5 (Dropout)              (None, 50)            0           activation_7[0][0]
  ____________________________________________________________________________________________________
  dense_3 (Dense)                  (None, 10)            510         dropout_5[0][0]
  ____________________________________________________________________________________________________
  activation_8 (Activation)        (None, 10)            0           dense_3[0][0]
  ____________________________________________________________________________________________________
  dropout_6 (Dropout)              (None, 10)            0           activation_8[0][0]
  ____________________________________________________________________________________________________
  dense_4 (Dense)                  (None, 1)             11          dropout_6[0][0]
  ====================================================================================================
  Total params: 981,819
  Trainable params: 981,819
  Non-trainable params: 0
  ____________________________________________________________________________________________________
