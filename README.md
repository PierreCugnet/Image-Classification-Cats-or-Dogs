# Image-Classification-Cats-or-Dogs
In this repository, you'll find how to create a CNN to perform binary classification problem on images using Keras library.
The dataset can be found on Kaggle at : https://www.kaggle.com/tongpython/cat-and-dog

CNN Architecture :

The CNN is composed by the following layers :  

  - Conv2D : 32 filters of size 3x3 using a ReLU function to remove linearity. We choose and input shape of 64x64x3 arrays (3D array to keep colors)
  - MaxPooling : Reduce the size of feature map --> reduce the processing (time complexity) while keeping the important info (keeping highest values) without reducting performance
  
  - Conv2D : same as above but we don't need to provide the input shape
  - MaxPooling :same as above

Now we are going to build a classic ANN to perform prediction

  - Flatten : We need to flatten the array provided by the 2 upper layers so we can use it as input for the ANN
  - Dense : ANN layer with 128 unit with relu activation function
  - Dense : ANN output layer with 1 unit and sigmoid(Sigmoid is used for binary outcome, use softmax for categorical)
  
Now we have a stacked CNN ready to be used ! 

We just need to compile : Optimizer='adam'  loss='binary_crossentropy', metrics='accuracy'

