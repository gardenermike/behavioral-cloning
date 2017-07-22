# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

This is a working Keras implementation of a model that learns to steer
based on the images exported from a simulator.

**Detailed description**
[Udacity](https://www.udacity.com/) provides a simulator for their Self-Driving Car nanodegree program.
The simulator uses the Unity engine to provide physics and rendering for a car to drive on multiple tracks. It also supports two other features: a series of screenshots from the perspective of the car, including a center, left, and right camera, can be recorded and saved with a .csv manifest file mapping the images to steering angles and throttle. Secondly, the simulator can call out to a service using websockets, and will pass along center camera screenshots and current speed information, and will listen for steering and throttle data to allow for remote driving.

This project is an [implementation](https://github.com/gardenermike/behavioral-cloning/blob/master/model.py) using [Keras](https://keras.io/) and [TensorFlow](https://www.tensorflow.org/) to autonomously drive the simulator using the raw image data.

Note that this implementation can be seen as a regression problem of images to steering angle. I experimented some with a recurrent layer in the network, but there was a strong tendency to overfit and just drive straight, so that is a project for another day.


[//]: # (Image References)

[Car screenshot]: ./images/center_2017_07_19_14_59_55_829.jpg "Example data"
[Narrow image]: ./images/output.jpg "Narrow slice of image"
[Augmented image 1]: ./images/output_0.jpg "Augmented image 1"
[Augmented image 2]: ./images/output_1.jpg "Augmented image 2"
[Augmented image 3]: ./images/output_2.jpg "Augmented image 3"
[Augmented image 4]: ./images/output_3.jpg "Augmented image 4"
[Augmented image 5]: ./images/output_4.jpg "Augmented image 5"
[Augmented image 6]: ./images/output_5.jpg "Augmented image 6"
[Augmented image 7]: ./images/output_6.jpg "Augmented image 7"
[Augmented image 8]: ./images/output_7.jpg "Augmented image 8"
[Augmented image 9]: ./images/output_8.jpg "Augmented image 9"
[Augmented image 10]: ./images/output_9.jpg "Augmented image 10"

---
### Files

The following files should be considered relevant
* model.py contains the model used to train on the first, simpler track
* model-track2.py contains the deeper model used for the second track
* drive.py provides support to drive the car in autonomous mode, including the websocket server, and generation of steering angles using the trained model. This file is mostly straight from Udacity's sample code, but I added a couple of features:
  - The simulator tends to lock up and quit accepting throttle commands. A quick "tap on the brake" fixes the problem, so my drive.py will push a small negative throttle value if the speed reported from the simulator drops below 0.5 mph.
  - I added support to run multiple models in parallel to get an ensemble value. I did not end up using the feature much, but it was interesting to experiment with, and could potentially be used to get smoother driving.
* model.h5 contains the trained model and weights for the first track.
* model-track2.h5 contains the trained model and weights for the second track
* architecture.pdf contains a detailed (zoomable!) description of the model I used on the second track.


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

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
