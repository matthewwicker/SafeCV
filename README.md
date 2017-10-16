# SafeCV
## Vision based algorithms for black-box falsification and safety testing of convolutional neural networks

SafeCV is mainly concerned with the falsification of deep, feed-forward convolutional neural networks. The package requires openCV, Keras, numpy and pomegranate. Running the examples requires matplotlib in addition. 

Installation with:
``` 
pip install SafeCV
```

As of right now, the package contains two main algorithms:

* DFMCS [Depth First Monte-Carlo Search] - A single monte-carlo based manipulation simulation based on human perception
* MCTS [Monte-Carlo Tree Search] - A monte-carlo tree search method for creating robust adversarial examples

Later, we will include a two-player game formulation for studying MNIST and CIFAR10 networks.

## Usage

Each run of DFMCS and MCTS must first initialize parameters: 

```
params_for_run = MCTS_Parameters(image, class, model)
```

These parameters can be changed to fit the desired performance of the algorithm. Then, the algorithm can be run with:

```
best_image, sev, prob, statistics = MCTS(params_for_run)
```

Where:

* best_image is the best adversarial example that was found,
* sev is the L0 Severity of the adversarial example,
* prob is the softmax output corresponding to the best adversarial example,
* statistics is a tuple of different runtime statistics that help illucidate perfomance 

## Runtime Parameters

Finally, we give a brief documentation of what each of the parameters controls

* model - The Neural Network model to be queried 
* ORIGINAL_IMAGE - The unmodified copy of the image (implicitly protected)
* TRUE_CLASS - The expected classification
* manip_method - a method that takes in two variables (pixel value and a constant) and dictates how the input will be manipulated
* VISIT_CONSTANT - Number of manipulations to make per time step
* SIGMA_CONSTANT - Varience to use when formulating the saliency distribution
* X_SHAPE - size of the X dimension of the input
* Y_SHAPE - size of the Y dimension of the input
* predshape - how to reshape the input before feeding it to the network
* kp, des, r - Keypoint values returned from an OpenCV feature detector
* EPSILON - Constant to be fed into the manipulation method
* verbose - Determines if the user wants to see all of the runtime outputs in the console
* preprocess - User defined method to say how an image should be preprocessed (default is to reshape to predshape and return)
* predict - Method for predicting the class and probability of an input
* small_image - if image is less than 50x50
* inflation_constant - When small_image is true, how much should we inflate the input to get a good saliency distribution
* backtracking_constant  - How many pixels to remove at each backtracking step
