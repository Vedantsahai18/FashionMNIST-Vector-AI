# FasionMNIST-Vector-AI

Handwritten Digit Recognition using Convolutional Neural Networks in Python with Keras

## MNIST dataset:

Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. Zalando intends Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

### Content

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total.   

Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255.   

The training and test data sets have 785 columns.   

The first column consists of the class labels (see above), and represents the article of clothing. 

The rest of 784 columns (1-785) contain the pixel-values of the associated image.

There are 10 different classes of images, as following: 

* **0**: **T-shirt/top**;   
* **1**: **Trouser**;   
* **2**: **Pullover**;   
* **3**: **Dress**;
* **4**: **Coat**;
* **5**: **Sandal**;
* **6**: **Shirt**;
* **7**: **Sneaker**;
* **8**: **Bag**;
* **9**: **Ankle boot**.


## Code Requirements
python 3.x with following modules installed

1. numpy
2. seaborn
3. tensorflow-gpu
4. keras
5. opencv2
6. boto3
7. json
8. Flask
9. Pandas
10. Glob
11. dotenv

## Model Description

I used a **Sequential** model.
* The **Sequential** model is a linear stack of layers. It can be first initialized and then we add layers using **add** method or we can add all layers at init stage. The layers added are as follows:

* **Conv2D** is a 2D Convolutional layer (i.e. spatial convolution over images). The parameters used are:
 * filters - the number of filters (Kernels) used with this layer; here filters = 32;
 * kernel_size - the dimmension of the Kernel: (3 x 3);
 * activation - is the activation function used, in this case `relu`;
 * kernel_initializer - the function used for initializing the kernel;
 * input_shape - is the shape of the image presented to the CNN: in our case is 28 x 28
 The input and output of the **Conv2D** is a 4D tensor.
 
* **MaxPooling2D** is a Max pooling operation for spatial data. Parameters used here are:
 * *pool_size*, in this case (2,2), representing the factors by which to downscale in both directions;
 
 * **Conv2D** with the following parameters:
 * filters: 64;
 * kernel_size : (3 x 3);
 * activation : `relu`;
 
* **MaxPooling2D** with parameter:
 * *pool_size* : (2,2);

* **Conv2D** with the following parameters:
 * filters: 128;
 * kernel_size : (3 x 3);
 * activation : `relu`;
 
* **Flatten**. This layer Flattens the input. Does not affect the batch size. It is used without parameters;

* **Dense**. This layer is a regular fully-connected NN layer. It is used without parameters;
 * units - this is a positive integer, with the meaning: dimensionality of the output space; in this case is: 128;
 * activation - activation function : `relu`;
 
* **Dense**. This is the final layer (fully connected). It is used with the parameters:
 * units: the number of classes (in our case 10);
 * activation : `softmax`; for this final layer it is used `softmax` activation (standard for multiclass classification)
 

Then we compile the model, specifying as well the following parameters:
* *loss*;
* *optimizer*;
* *metrics*. 


## Execution

First unzip the mnist datset folder inside the input directory then 

To run the train code type,

`python train.py  --train <TRAIN DATA PATH> --test <TEST DATA PATH>`

example 
`python train.py  --train /input/fashion_mnist_train.csv --test /input/fashion_mnist_test.csv`

To run the prediction code on an custom image type,

`python predict.py  --image <PREDICT IMAGE DATA PATH>`

example

`python predict.py  --image /images/predict.png`

## Metrics

The best accuracy is obtained for Class 1, Class 5, Class 8, Class 9  and Class 7. Worst accuracy is for Class 6.   

The recall is highest for Class 8, Class 5 and smallest for Class 6 and Class 4.    

f1-score is highest for Class 1, Class 5 and Class 8 and smallest for Class 6 followed by Class 4 and Class 2.  

Let's also inspect some of the images. We created two subsets of the predicted images set, correctly and incorrectly classified.

Test loss: 0.2115551935195923
Test accuracy: 0.9232


## Update

### For running on GPU enabled devices:

Please uncomment the following line from **digit_recogniser.py** (line no. 70) file:
```
tfback._get_available_gpus = _get_available_gpus
```

**Note: If you are using the tensorflow 2.1, then you may get an error "AttributeError: module'tensorflow_core._api.v2.config' has no attribute 'experimental_list_devices'"**

As the experimental_list_devices is deprecated in tf 2.1. A simple snippet is injected into the code to make the code 

## Note

To upload the model metrics json and the model to the s3 bucket kindly provide the AWS access and secrret key to your respective bucket.For futher setup of Boto3 and how to use it kindly visit the documentation of Boto3