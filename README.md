# Facial Landmark Detection

<h3>Project Overview</h3>

![](https://github.com/parth1620/Facial-landmark-Detection/blob/master/Images/key_pts_example.png)

Facial keypoints (also called facial landmarks) are the small magenta dots shown on each of the faces in the image above. In each training and test image, there is a single face and **68 keypoints, with coordinates (x, y), for that face**.  These keypoints mark important areas of the face: the eyes, corners of the mouth, the nose, etc. These keypoints are relevant for a variety of tasks, such as face filters, emotion recognition, pose recognition, and so on. Here they are, numbered, and you can see that specific ranges of points match different portions of the face.

![](https://github.com/parth1620/Facial-landmark-Detection/blob/master/Images/landmarks_numbered.jpg)

<h3>Notebook 1 : Loading and Visualising Facial Keypoints Data</h3>

The first step in working with any dataset is to become familiar with your data; you'll need to load in the images of faces and their keypoints and visualize them! This set of image data has been extracted from the [YouTube Faces Dataset](https://www.cs.tau.ac.il/~wolf/ytfaces/), which includes videos of people in YouTube videos. These videos have been fed through some processing steps and turned into sets of image frames containing one face and the associated keypoints.

#### Training and Testing Data

This facial keypoints dataset consists of 5770 color images. All of these images are separated into either a training or a test set of data.

* 3462 of these images are training images, for you to use as you create a model to predict keypoints.
* 2308 are test images, which will be used to test the accuracy of your model.

The information about the images and keypoints in this dataset are summarized in CSV files, which we can read in using `pandas`. Let's read the training CSV and get the annotations in an (N, 2) array where N is the number of keypoints and 2 is the dimension of the keypoint coordinates (x, y).

Data-Set link - https://s3.amazonaws.com/video.udacity-data.com/topher/2018/May/5aea1b91_train-test-data/train-test-data.zip

<h3>Notebook 2 : Creating and Training Model</h3>

The first step was to load the data and to resize them in shape (224,224,1) as input to the neural network. As we resize all the images the final shape of training image data is (3462,224,224,1) and keypoints are also rescaled and final shape is (3462,68,2) which is then reshaped to (3462,136) as ouput nodes of neural network.

Images were normalized to convert image into grayscale values with range of [0,1] and keypoints were also normalized  with the range of [-1,1]

Model was created using tensorflow keras API.

**Model Architecture**

![](https://github.com/parth1620/Facial-landmark-Detection/blob/master/Images/landmark-model.png)


<h3>Notebook 3 : Facial Keypoint Detection</h3>

These notebook completes pipeline to detect face points on the face image.

First Haar-Cascade Face Detection is applied on the faces and co-ordinates from the detection (x,y,w,h) are used to crop the image after then resize to shape 224,224,1 as an input to the model.

The cropped and resized image is then passed to the model to predict keypoints.

![](https://github.com/parth1620/Facial-landmark-Detection/blob/master/Images/obamas.jpg)

![](https://github.com/parth1620/Facial-landmark-Detection/blob/master/Images/Detection.png)

![](https://github.com/parth1620/Facial-landmark-Detection/blob/master/Images/obamakeypoints.png)

![](https://github.com/parth1620/Facial-landmark-Detection/blob/master/Images/michelle.png)


<h3>Complete-pipeline.py</h3>

Video capture is used to capture frames and frames are input the haar cascade to detect face and then detected face are the inputs to the model and outputs keypoints.

![](https://github.com/parth1620/Facial-landmark-Detection/blob/master/Images/videocapture.gif)



