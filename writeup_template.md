#**Behavioral Cloning Project** 

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
![alt text][image1]

####2. Attempts to reduce overfitting in the model

The Model has been heavily reguralized using Dropout and L2 regularization. Since the total parameters that are trained is 981,819, and dataset I have used is highly skewed to center steering there are higher chances of overfitting. 
This model works fine in the simulator. Given the interest of time and limitation on the VARIETY of data used, I am sure this can be further improved even with less trainable parameters.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 164).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. The udacity provided was hevily skewed to center driving. I have tried to encorporate generate driving ddata for recovery from road shoulders.

##### Udacity Provided Dataset with Left Right Augmentation:
![alt text][image2]


##### Dataset with new Vehicle Recovery from Shoulders
![alt text][image3]

For details about how I created the training data, see the next section. 
