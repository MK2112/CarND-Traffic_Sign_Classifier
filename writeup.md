
# **Traffic Sign Recognition**

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./writeup-images/picked_signs.PNG "Traffic Sign Examples"
[image2]: ./writeup-images/distribution_graphs.PNG "Visualization"
[image3]: ./writeup-images/1.jpg "Speed limit (30kmh)"
[image4]: ./writeup-images/12.jpg "Priority road"
[image5]: ./writeup-images/13.jpg "Yield"
[image6]: ./writeup-images/15.jpg "No vehicles"
[image7]: ./writeup-images/39.jpg "Keep left"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

- The project submission includes all required files <br>(Ipython notebook with code, HTML output of the code, A writeup report in either pdf or markdown).
- The submission includes a basic summary of the data set.
- The submission includes an exploratory visualization on the dataset.
- The submission describes the preprocessing techniques used and why these techniques were chosen.
- The submission provides details of the characteristics and qualities of the architecture, including the type of model used, the number of layers, and the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.
- The submission describes how the model was trained by discussing what optimizer was used, batch size, number of epochs and values for hyperparameters.
- The submission describes the approach to finding a solution. Accuracy on the validation set is 0.93 or greater.
- The submission includes five new German Traffic signs found on the web, and the images are visualized. Discussion is made as to particular qualities of the images or traffic signs in the images that are of interest, such as whether they would be difficult for the model to classify.
- The submission documents the performance of the model when tested on the captured images. The performance on the new images is compared to the accuracy results of the test set.
- The top five softmax probabilities of the predictions on the captured images are outputted. The submission discusses how certain or uncertain the model is of its predictions.
---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! Thank you for doing so :)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used python functionalities to calculate summary statistics of the traffic
signs data set:

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **(32, 32, 3)**
* The number of unique classes/labels in the data set is **43**

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. I pulled ten random images.

![alt text][image1]

I then went on to visualize the respective distributions of examples per training-, test- and validation-set.

![alt text][image2]

For each of these data sets the distribution is imbalanced. This might imply that the data sets should be enriched by additional examples for underrepresented traffic signs.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I normalized the image pixel data using the formula `(image - 128.) / 128.`.<br> Admittedly I took some extra time to get my head around this particular concept in detail. By doing so I decided to proceed without the optional steps in order to see how this normalization concept would contribute to the network's performance (on training but then on unknown images aswell). This would later show to have a good learning effect for me.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model was heavily inspired by the original LeNet architecture. It consists of the following layers:

| Layer                 |     Description                               |
|:---------------------:|:---------------------------------------------:|
| Input                 | 32x32x3 RGB image                             |
| Convolution #1 5x5    | 1x1 stride, valid padding, outputs 28x28x6    |
| RELU                  |                                               |
| Dropout               | probability of keeping: 0.9                   |
| Max pooling           | 2x2 stride, outputs 14x14x6                   |
| Dropout               | probability of keeping: 0.9                   |
| Convolution #2 5x5    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU                  |                                               |
| Dropout               | probability of keeping: 0.9                   |
| Max pooling           | 2x2 stride, outputs 5x5x16                    |
| Dropout               | probability of keeping: 0.9                   |
| Flattening            | takes 5x5x16, outputs 400                     |
| Fully connected #1    | outputs 120                                   |
| RELU                  |                                               |
| Dropout               | probability of keeping: 0.9                   |
| Fully connected #2    | outputs 84                                    |
| RELU                  |                                               |
| Dropout               | probability of keeping: 0.9                   |
| Fully connected #3    | outputs 43                                    |
| Softmax               |                                               |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer to minimize loss function. In order for this to work I defined (and admittedly played with) the values of these hyperparameters:

```
# This many images are fed to the net per individual training operation
batch_size = 128

# Times to iterate through training set
# Found this to be performing better than 40, 30 etc.
# while still trying cautiously to avoid overfitting
epochs = 50

# Learning rate, meaning rate of updating network parameters
rate = 0.001

# Probability of keeping weight while applying dropout
prob_keep = 0.9

# Standard deviation (tensorflow calls it 'stddev')
sigma = 0.1
```

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

After quite an amount of tries involving parameter tuning (this resulting in the values shown above) my final model results were:
* training set accuracy of **1.000**
* validation set accuracy of **0.953**
* test set accuracy of **0.947**

An iterative approach was chosen to achieve these results.
* What was the first architecture that was tried and why was it chosen?
  * The architecture design from the start was based on LeNet. My first iteration however was too minimalistic of an approach. It lacked Dropouts and the first two of the fully connected layers.
* What were some problems with the initial architecture?
  * This 'too minimalistic' approach had the effect of really showing me directly what dropout means. It helps to diffuse experience more evenly across the network. The lack of this meant great results on the training set, remarkably poorer results on the validation set, terrible results on the test set. In short: Overfitting.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
  * I introduced two components. #1: Dropout after every MaxPooling or applied RELU and #2: Two additional, fully connected layers. This - I thought - would create a bigger set of weights (I call them 'knobs and dials') that could be affected through training experience. Now the dropouts distributed the experience across these 'knobs and dials' more evenly. Hence the root of the 'painful overfitting' problem luckily got addressed.
* Which parameters were tuned? How were they adjusted and why?
  * The learning rate was lowered from `0.005` to `0.001` and the variable `prob_keep` was introduced. I lowered its initial value of `0.95` to `0.9` which admittedly is still relatively high, but the performance numbers seemed good with this setting already.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
  * Most important design choices include the use of convolutional layers aswell as dropout.
  * Convolutional layers provide a very much needed way of giving the natually rather informationally flat (3 layers of information in an RGB image but rather big width and height) more informational depth while being able to reduce two-dimensional width and height. This makes the image's motive decouple from the image itself, as the motive's position in the image becomes less relevant to the content identifying process. This means applying convolutional layers enables the network to better focus on the images content (that being traffic signs in this example) rather than its positioning in the image.
  * Dropout prevents from smaller subsets of weights of the network to become overly determining and influential in the networks making of guesses. Once applied they make sure to 'turn off' a certain percentage of random neurons during training. This causes a more equal distribution of trained experience / weight-updating. 'Knowledge' is distributed -per training run- to random subsets of neurons. This can help avoid overfitting. The performance of the network on unknown data can increase from applying dropout correctly.

If a well known architecture was chosen:
* What architecture was chosen?
  * Yann LeCun's LeNet served as the basis for this neural network. LeNet is a Convolutional Neural Network.
* Why did you believe it would be relevant to the traffic sign application?
  * As I unterstood it, CNNs address the problem of identifying an image's motive while also addressing that e.g. positioning and three-dimensional warping of the motive do not change the motive's 'character'. Images that display the same thing from different angles get the CNN therefore to still identify this thing. So traffic signs from different angles are still traffic signs and the CNN is still capable of identifying the motive and specify which traffic sign means what.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
  * Given a validation set accuracy of 95.3% and a test set accuracy of 94.7% the network can take well educated guesses on traffic signs it has never gathered experience from (during training). This -and specifically the result with the test set- implies that the network generalized information gathered from *other* traffic signs rather than learning the training images by heart (Overfitting). If it had done that, the performance on the test set and the validation set aswell would have been significantly lower.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image3]<br>
The image might be clear but the motive is tilted.

![alt text][image4]<br>
This image -compared e.g. to the first image- lacks contrast. The house in the lower right corners introduces a set of edges and structures that the network has to 'know' to ignore.

![alt text][image5]<br>
This image's motive has realtively strong tilt.<br>

![alt text][image6]<br>
This I chose to be an extreme of the previous image, introducing an even harder tilt.<br>

![alt text][image7]<br>
A more frontal shot but containing colors and patterns that could be misidentified as being part of a different traffic sign.<br>

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Label			        |     Network	        					|
|:---------------------:|:---------------------------------------------:|
| 30 km/h      		| 30 km/h   									|
| Priority road     			| Priority road 										|
| Yield					| Yield											|
| No vehicles	      		| No vehicles					 				|
| Keep left			| Keep left      							|

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.7%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 3rd to last cell of the Jupyter notebook.
These were the probabilities for images 1 to 5:

![alt text][image3]<br>

For the first image, the model is very sure that this is a 30 km/h speed limit sign which it was in fact. The network made this classification with a high level of confidence. The top five soft max probabilities look like this:

| Probability     | Sign ID | Prediction            |
|:---------------:|:-------:|:---------------------:|
| 9.83 * 10^(-1)  | 1       | Speed limit (30 km/h) |
| 1.69 * 10^(-2)  | 0       | Speed limit (20 km/h) |
| 9.90 * 10^(-7)  | 2       | Speed limit (50 km/h) |
| 1.26 * 10^(-7)  | 35      | Ahead only            |
| 4.51 * 10^(-8)  | 5       | Speed limit (80 km/h) |

![alt text][image4]<br>

For the second image which is the priority road sign the network determined it to be a priority road sign correctly. It this time determined with a much higher confidence:

| Probability     | Sign ID | Prediction            |
|:---------------:|:-------:|:---------------------:|
| 1.00            | 12      | Priority road |
| 4.15 * 10^(-13) | 26      | Traffic signals |
| 4.27 * 10^(-14) | 42      | End of no passing by vehicles over 3.5 metric tons |
| 1.37 * 10^(-14) | 10      | No passing for vehicles over 3.5 metric tons |
| 5.05 * 10^(-15) | 38      | Keep right |

![alt text][image6]<br>

For the third image which is the no vehicles road sign the network determined it to be a no vehicles road sign correctly. It this time determined the motive with 95.3% confidence:

| Probability     | Sign ID | Prediction            |
|:---------------:|:-------:|:---------------------:|
| 9.53 * 10^(-1)  | 15      | No vehicles           |
| 2.07 * 10^(-2)  | 1       | Speed limit (30 km/h) |
| 1.49 * 10^(-2)  | 25      | Road work             |
| 4.29 * 10^(-3)  | 2       | Speed limit (50 km/h) |
| 3.09 * 10^(-3)  | 4       | Speed limit (70 km/h) |

![alt text][image7]<br>

For the fourth image which is the keep left road sign the network determined it to be a keep left road sign correctly. It this sign was the one of the bunch that the network made the lowest confidence choice, althougth it was correct:

| Probability     | Sign ID | Prediction            |
|:---------------:|:-------:|:---------------------:|
| 8.55 * 10^(-1)  | 39      | Keep left             |
| 1.41 * 10^(-1)  | 31      | Wild animals crossing |
| 2.41 * 10^(-3)  | 1       | Speed limit (30 km/h) |
| 1.29 * 10^(-3)  | 22      | Bumpy road            |
| 1.04 * 10^(-4)  | 29      | Bicycles crossing     |

![alt text][image5]<br>

The final image displayed a yield sign. The network determined the image to show the yield sign with a confidence of 100%.

| Probability     | Sign ID | Prediction            |
|:---------------:|:-------:|:---------------------:|
| 1.00            | 13      | Yield                 |
| 6.15 * 10^(-10) | 5       | Speed limit (80km/h)  |
| 4.17 * 10^(-10) | 38      | Keep right            |
| 1.75 * 10^(-10) | 3       | Speed limit (60km/h)  |
| 1.34 * 10^(-10) | 12      | Priority road         |

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
