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
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

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


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I took iterative approach to come to final model. First, I tried with grayscale images and could only get upto 93% accuracy. Same was the case with Y channel. With RGB, I could get slightly better results going upto 94%. I also tried different epochs and learning rate. I found that higher learning rate reaches desired accuracy faster but does not give stable numbers. Here's comparison of validation accuracy graph vs epochs for learning rate of 0.01 on the left and learning rate of 0.002 on the right.

![alt text][image5] | ![alt text][image6]

Looking at the training accuracy I saw that there was gap in training and validation accuracies. Training accuracy was easily reaching > 99% within few epochs. This indicated that the model was being overfitted to the training data and was not generalizing well to the validation data. So I decided to try out regularization technique to prevent overfitting. I tried Dropout with `keep_prob=0.5` which proved very beneficial and increased the accuracy to >95%. Here's my training and validation accuracies vs epochs:

![alt text][image7] | ![alt text][image8]

Here's validation accuracy curve vs epochs for final model:

![alt text][image9]

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


