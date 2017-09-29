# **Traffic Sign Recognition** 

## Writeup Template

### This is a writeup of Project 2: Build a Traffic Sign Classifier of Udacity's Self Driving Car Nanodegree.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/dataset_exploration.PNG "Dataset Exploration"
[image2]: ./output_images/class_histogram.PNG "Class Histogram"
[image3]: ./output_images/preprocessing_grayscale.PNG "Preprocessing Grayscale"
[image4]: ./output_images/preprocessing_y_channel.PNG "Preprocessing Y Channel"
[image5]: ./output_images/y_256_20_01.PNG "Validation Accuracy Graph for Higher Learning Rate"
[image6]: ./output_images/gray_256_20_002.PNG "Validation Accuracy Graph for Lower Learning Rate"
[image7]: ./output_images/rgb_256_20_002_training.PNG "Training Accuracy Curve without Dropout"
[image8]: ./output_images/rgb_256_20_002_training_dropout.PNG "Training Accuracy Curve of Final Model"
[image9]: ./output_images/rgb_256_20_002_validation_dropout.PNG "Validation Accuracy Curve of Final Model"
[image10]: ./output_images/test_images2.PNG "Validation Accuracy Curve of Final Model"
[image11]: ./output_images/bicycles_crossing.PNG "Bicycles Crossing"
[image12]: ./output_images/slippery_road.PNG "Slippery Road"
[image13]: ./output_images/beware_of_ice_snow.PNG "Beware of Ice and Snow"
[image14]: ./output_images/dangerous_curve_right.PNG "Dangerous Curve Right"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my IPython notebook [Traffic Sign Classifier.ipynb](https://github.com/kambliketan/Self_Driving_Car_Traffic_Sign_Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library in code cell 1 and 2 to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799 images
* The size of the validation set is 4,410 images
* The size of test set is 12,630 images
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

In code cells 3, 4, and 5 of the IPython notebook, I provide code to visualize and explore the dataset. Here is an exploratory visualization of the data set.

![alt text][image1]

Next, I show the histogram of number of images belonging to each of the 43 classes for all 3 datasets i.e. training, validation, and testing dataset. It can be seen that although the numbers differ in each set, the shape of the histogram is preserved. But it can be noted that there is an uneven class representation.

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because most of the features are about shapes of traffic signs rather than the color. I used `cv2.cvtColor()` function to convert the images to grayscale. The code is shown in code cell 6 in `convert_dataset_to_gray` function.

Here are few images after conversion to grayscale:

![alt text][image3]

Next, I normalized the image data by employing Min-Max scaling scheme in `normalize` function in code cell 9. Normalizing makes the data well conditioned.

Parallely on separate effort, in code cell 8 I tried Y channel of the YCrCb color space representation of the image and RGB 3 channel representation of the images. In code cells 10, 11, 12 I normalized all three feature sets. Here're Y channel images:

![alt text][image4]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I explored LENET model that was discussed in lessons. My final model consisted of the following layers and is shown in code cell 18 of the IPython Notebook:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x16    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Fully connected		| input 400, output 120.        			    |
| Dropout               |                                               |
| Fully connected		| input 120, output 84.        			        |
| Dropout               |                                               |
| Fully connected		| input 84, output 43.        			        |
| Softmax				|           									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I took iterative approach to come to final model. First, I tried with grayscale images and could only get upto 93% accuracy. Same was the case with Y channel. With RGB, I could get slightly better results going upto 94%. I also tried different epochs and learning rate. I found that higher learning rate reaches desired accuracy faster but does not give stable numbers. Here's comparison of validation accuracy graph vs epochs for learning rate of 0.01 on the left and learning rate of 0.002 on the right.

![alt text][image5] | ![alt text][image6]

Looking at the training accuracy I saw that there was gap in training and validation accuracies. Training accuracy was easily reaching > 99% within few epochs. This indicated that the model was being overfitted to the training data and was not generalizing well to the validation data. So I decided to try out regularization technique to prevent overfitting. I tried Dropout with `keep_prob=0.5` which proved very beneficial and increased the accuracy to >95%. Here's my training and validation accuracies vs epochs:

![alt text][image7] | ![alt text][image8]

Here's validation accuracy curve vs epochs for final model:

![alt text][image9]

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 98.98%
* validation set accuracy of 94.14% 
* test set accuracy of 91.1%

I have described the approach in above points.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I chose to test on:

![alt text][image10]

The first image `Speed limit (30km/h)` can be tricky to exactly nail the speed limit i.e. whether its 30, 50 or 80, but other than that the image looks clear, bright and hence straight forward. This fact is reflected in the next section where I show the top five predictions, along with the softmax probability of how confident the classifier is. The classifier has 99.74% sure and makes correct prediction in this case. 

The second image `Ahead only` has a some graffity on it that can make things a bit trickier but other than that the image looks clear, bright and hence straight forward. And the classifier makes no mistake there. 

Third image `End of no passing` is dimly lit and not very clear and resembles many other traffic signs, see forth image above for example. So it is easy to make mistake here.

Fourth image `No passing` has low brightness and resembles some other classes like the third image above. 

Fifth example `No entry` is bright and clear and classifier should not have trouble classifying it. And indeed as shown in next section, it is almost 100% confident about its assertion.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)     | Speed limit (30km/h)   							| 
|  Ahead only  |  Ahead only 							|
| End of no passing			| End of no passing									|
| No passing	| No passing					 		|
| No entry	| No entry  |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located at the bottom of the Ipython notebook.

Here are the top 5 predictions for each of the test images. As we can see the model's first five predictions are very very intuitional! (Magic:))

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.997478.             | Speed limit (30km/h)                          |
| 0.00250469.           | Speed limit (20km/h)                          |
| 8.5459e-06.           | Speed limit (50km/h)                          |
| 8.0522e-06.           | Speed limit (70km/h)                          |
| 1.032e-07.            | Speed limit (80km/h)                          |

Second Image:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0.                  | Ahead only                                    |
| 4e-10.                | Turn right ahead                              |
| 2e-10.                | Go straight or left                           |
| 1e-10.                | Keep left                                     |
| 0.0.                  | Yield                                         |

Third Image:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.999938. | End of no passing |
| 3.85629e-05. | End of all speed and passing limits |
| 2.03703e-05. | Vehicles over 3.5 metric tons prohibited |
| 1.8229e-06. | No passing |
| 8.049e-07. | Go straight or right |

Fourth image:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.999992. | No passing |
| 7.0573e-06. | Vehicles over 3.5 metric tons prohibited |
| 7.686e-07. | No passing for vehicles over 3.5 metric tons |
| 5.855e-07. | Dangerous curve to the right |
| 9.7e-09. | End of no passing |

Fifth image:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0. | No entry |
| 0.0. | No passing |
| 0.0. | Stop |
| 0.0. | Dangerous curve to the right |
| 0.0. | Priority road |

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


