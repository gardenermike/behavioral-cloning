# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

![screenshot][screenshot]

This is a working Keras implementation of a model that learns to steer
based on the images exported from a simulator.

**Detailed description**

[Udacity](https://www.udacity.com/) provides a [simulator](https://github.com/udacity/self-driving-car-sim) for their Self-Driving Car nanodegree program.
The simulator uses the Unity engine to provide physics and rendering for a car to drive on multiple tracks. It also supports two other features: a series of screenshots from the perspective of the car, including a center, left, and right camera, can be recorded and saved with a .csv manifest file mapping the images to steering angles and throttle. Secondly, the simulator can call out to websocket service, passing center camera screenshots and current speed information, and listening for steering and throttle data to allow remote driving.

An example output image from the simulator is below:

![Car screenshot][Car screenshot]

This project is an [implementation](https://github.com/gardenermike/behavioral-cloning/blob/master/model.py) using [Keras](https://keras.io/) and [TensorFlow](https://www.tensorflow.org/) to autonomously drive the simulator using the raw image data.

This implementation can be seen as a regression problem of images to steering angle. I experimented some with a recurrent layer in the network, but there was a strong tendency to overfit and just drive straight, so that is a project for another day.


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
[gif_as_model]: ./images/track_2_as_model.gif "Track 2 as model"
[screenshot]: ./images/screenshot.png "Screenshot"
[left_image]: ./images/left_2017_06_23_12_57_27_008.jpg "Left image"
[center_image]: ./images/center_2017_06_23_12_57_27_008.jpg "Center image"
[right_image]: ./images/right_2017_06_23_12_57_27_008.jpg "Right image"

---
### Files

The following files should be considered relevant:
* model.py contains the model used successfully on both tracks
* model-simple.py contains a smaller model used successfully on the basic track
* drive.py provides support to drive the car in autonomous mode, including the websocket server and generation of steering angles using the trained model. This file is mostly straight from Udacity's sample code, but I added a couple of features:
  - The simulator tends to lock up and quit accepting throttle commands. A quick "tap on the brake" fixes the problem, so my drive.py will push a small negative throttle value if the speed reported from the simulator drops below 0.5 mph.
  - I added support to run multiple models in parallel to get an ensemble value. I did not end up using the feature much, but it was interesting to experiment with, and could potentially be used to get smoother driving.
  - For the second track, the [speed](https://github.com/gardenermike/behavioral-cloning/blob/master/drive.py#L48) should be set to 5 instead of 9 mph, as that corresponds better to my cautious driving on that track.
* model.h5 contains the trained weights for the first track.
* model-track2.h5 contains the trained weights for the second track
* architecture.pdf contains a detailed (zoomable!) description of deeper the model I used successfully on both tracks

To run a saved model, run:
`python drive.py [model_weights_file]`


### Model Architecture and Training

#### 1.Model architecture

I used a deep convolutional network. For a zoomable diagram of the deeper architecture I used successfully on both tracks, [check out the pdf](https://raw.githubusercontent.com/gardenermike/behavioral-cloning/master/architecture.pdf).

The model could be considered loosely modeled on the LeNEt architecture, with additional layers added. With the idea that model depth represents abstraction, I opted for a deeper rather than wider model. I tinkered with the model quite a bit to balance performance and size, and to get solid performance that would generalize across both tracks.
I used [elu](https://arxiv.org/abs/1511.07289) activation functions on each layer.
I spent a number of iterations adjusting filter count in the convolutions, finding a quick-performing and reasonably small model that still learned successfully. The size of the first fully-connected layer turned out to be the biggest factor in the model size, which reminded me how much advantage convolutional layers provide by using shared weights.
I also experimented with variations of max pooling. Max pooling lowered performance somewhat but dramatically sped up training by reducing the size of the model. The final architecture is careful with max pooling: it is applied three times, at carefully spaced intervals, and in addition, two of my max pooling layers pool only in the horizontal dimension, as the source images contain more crucial and limited vertical than horizontal information.

One thing to note in the model is the severe cropping: the vast majority of the image has been removed. In fact, I found that I could only successfully train the model with such severe cropping. In particular, any part of the image suggesting the _future_ was problematic. Since the autonomous driving process with the simulator works on a per-image basis with no state, each image needs to stand alone. I found that removing most of the upcoming view of the road ahead allowed the model to "live in the moment". The driving was a little choppier, but odd artifacts like a rock in front of a lane line would not confuse the model. On the second track, cropping much of the horizontal data (100 pixels on each side) works well, as the center dotted line is the best indicator of correct position.

I experimented with color quite a bit. In the lambda preprocessing layer, I am currently using Tensorflow inside the model to convert RGB to HSV, then [dropping all but the S(aturation) layer](https://github.com/gardenermike/behavioral-cloning/blob/master/model.py#L202). I found the the S channel alone was not only sufficient but superior. I struggled with many iterations of the model to deal with the entrance to the dirt road on the first track. The S channel captured the boundary of the paved road best, allowing the model to see the corner properly.
I performed grander experiments on the second track. I actually was able to drive most of the track with just one [carefully crafted channel](https://github.com/gardenermike/behavioral-cloning/blob/master/model-track2.py#L217).
I made a gif from the perspective of the model using my custom channel:

![Track 2 from the perspective of the car, with custom channel][gif_as_model]

In the end, I switched back to a more standard use of colors in interest of trying to learn something more generalizable that would work on both tracks. I found that using all of the R, G, B, H, S, and V channels led to quicker convergence on the second track. If using my code to train the second track, I recommend commenting out [line 202](https://github.com/gardenermike/behavioral-cloning/blob/master/model.py#L202) to use all six color channels.

I also tried [separable convolutions](https://arxiv.org/abs/1610.02357) in addition to standard convolutions. I found that they did improve my model performance if used in layers after the first layer, since my first layer was already carefully crafted.

I also used batch normalization in a couple of layers. Batch normalization caused a much smoother decline in the loss. Interestingly, it caused the validation loss to drop much more slowly than the training loss, but the validation loss eventually dropped to lower than the training loss. Such behavior is backwards from what I'd usually expect in training, where the model eventually overfits. My generous use of dropout and image augmentation seemed to prevent overfitting, and the batch normalization just helped clarify when my model actually had generalized.

#### 2. Avoiding overfitting

A lot of my work was around getting data that dealt with _literal_ corner cases in the data :). Loss could be driven fairly low by a model that just refused to turn at all, so capturing strong turns required good data.

My first tactic was to use a [tunable exponent approach](https://github.com/gardenermike/behavioral-cloning/blob/master/model.py#L130) to strongly prefer training data with a strong steering angle. The model was far more exposed to sharp turns than straight lines. I tuned the exponent and leak quite a bit to get a good balance on the first track.

After struggling to get good performance on the second track, I added some data augmentation to every training image. I started with random shadows on the image, which worked well in my customized color channel, but completely derailed the model on standard RGB data. The real success, though, was in a [shear function I worked up](https://github.com/gardenermike/behavioral-cloning/blob/master/model.py#L98) to rotate the top of the image left or right, leaving the base of the image static. Adding the shear provided an essentially infinite variety of steering situations with accurate steering angles, allowing my model to generalize far better. Here is a sampling of augmented images, each from the same source training image:

![Augmented image 1][Augmented image 1]
![Augmented image 2][Augmented image 2]
![Augmented image 3][Augmented image 3]
![Augmented image 4][Augmented image 4]
![Augmented image 5][Augmented image 5]
![Augmented image 6][Augmented image 6]
![Augmented image 7][Augmented image 7]
![Augmented image 8][Augmented image 8]
![Augmented image 9][Augmented image 9]
![Augmented image 10][Augmented image 10]


Adding the random shear was really key to getting around the very challenging second track. Going back and revisiting the first track with the shear augmentation, I found that it was sufficient to get good steering there as well. As a sidenote, I started with angles between -1/2 and 1/2 pi, which the model learned, but were excessive, as the maximum steering angle on the car was 0.25. I got better convergence with values within the usable range.

In addition to the shadow and shear augmentation, I also trained on every image flipped horizontally (with a negated steering angle) as well as unflipped to avoid a turning bias. I also added images from the left and right cameras (with a 0.25 radian angle added) on 20% of the rows to capture scenes out-of-center without having to drive unsafely. Those side images allowed the model to recover after accidentally drifting off center. The videos show a couple of cases where the car neared the edge of the road and then swerved back to center. The side-camera images really make a huge difference in the smoothness of the model and recovery from drifts to the side.

Here are some camera images from the left, center, and right cameras:

![Left image][left_image]
![Center image][center_image]
![Right image][right_image]


#### 3. Model parameter tuning

The model used an adam optimizer, so not much tuning was needed outside of fitting the model to have weights less than 100MB, which was my target to allow upload to github. Accidentally surpassing that target was surprisingly easy.

#### 4. Appropriate training data

I found that I needed to drive slowly and well-centered on the track to get good data, followed by adding additional data for areas (corners, typically, and the spot with parallel roads on the second track) that I found problems in with a model trained on the first pass of generated data.

For both tracks, I was able to learn to drive with data from only three loops around the track (with extra at corners), together with data augmentation described above.


### Retrospective
I spent a lot of time on this project, especially on the second track. I might have a hundred hours or more spent training iterations of model architecture, color variations, cropping, and so on.
Key observations are:
* Less is more. Extreme cropping, especially vertically, gave me better results.
* Augmentation was key. There was no way I could get enough data by driving on my own. Augmenting, especially with shear added, made a tremendous difference.
* Color matters. I was never fully successful with RGB data only. I suspect that I _may_ have been able to pull it off with my final architecture, but HSV data did much better than RGB in my training, and typically one channel was best.

### Video!

.mp4 files from the perspective of the vehicle are included for both tracks in this repository. As a shortcut, though, you can hop on over to YouTube with the two links below.

[![Track 1](http://img.youtube.com/vi/7yL9rPkTVy8/hqdefault.jpg)](https://youtu.be/7yL9rPkTVy8)

[![Track 2](http://img.youtube.com/vi/bSAa5H7R92s/hqdefault.jpg)](https://youtu.be/bSAa5H7R92s)
