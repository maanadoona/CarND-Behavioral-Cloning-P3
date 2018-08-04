# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/centerlane/center_2018_08_02_15_51_34_674.jpg "Center Image"
[image2-1]: ./examples/centerlane/left_2018_08_02_15_51_34_674.jpg "Left Image"
[image2-2]: ./examples/centerlane/right_2018_08_02_15_51_34_674.jpg "Right Image"
[image3]: ./examples/recoverylane/center_2018_08_02_15_54_42_364.jpg "Center Recovery Image"
[image4]: ./examples/recoverylane/left_2018_08_02_15_54_42_364.jpg "Left Recovery Image"
[image5]: ./examples/recoverylane/right_2018_08_02_15_54_42_364.jpg "Right Recovery Image"
[image6]: ./examples/flip/center_flip_2018_08_02_15_51_34_674.jpg "Center Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.
I drived the car of two tracks. At First track I made an effort to go through the center and at last track, I moved the car close to left line or right line.
Additionally, I got the average velocity from 15 to 20. At the first time I drive as fast as I can. But when I trained the networks, I got the poor result which the car was swung left and right eventually was jumped the track to outside.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to get the same result with nvidia's model.

My first step was to use a convolution neural network model similar to the lenet I thought this model might be appropriate because it is well-known and is rated as standard.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that dropout function is added.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track like the entrance of bridge and close to river. To improve the driving behavior in these cases, I moved the car slowly to get many images in those cases.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 57-69) consisted of a convolution neural network with the following layers and layer sizes same as nvidia's but I modified the size to mine. and I add the Dropout function to prevent overfitting.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]
![alt text][image2-1]
![alt text][image2-2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recovery angles. These images show what a recovery looks like starting from outsie to inside :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image2]
![alt text][image6]

After the collection process, I had X number of data points. I then preprocessed this data by cropping.
Raw images are burden to process, Cropping images help to reduce time for learning.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by seeing loss value was extremly reduced. I used an adam optimizer so that manually training the learning rate wasn't necessary.
