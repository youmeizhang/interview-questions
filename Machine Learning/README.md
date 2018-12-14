## Machine Learning Interview Questions
##### 1. Neural network
it is a connection of many very tiny processing elements called as neurons

##### 2. Why neural network
adaptive learning, self-organization, real-time operation

##### 3. Disadvantage
They require a large amount of training data and they are not strong enough to work in real world. <br >
(Practical applications most employ supervised learning)

##### 4. Unsupervised learning in NN
perform some kind of data compression, such as dimensionality reduction or clustering

##### 5. Activation function
The goal of activation function is to introduce non-linearity into the neural network so that it can learn more complex function. Without it, the network would be only able to learn function which is a linear combinations of its input data.
* Neuron: given input it calculates a “weighted sum” of its input and add a bias and then decide whether it should be fired or not. Activation function does this work for the network: to check the Y value produced by a neuron and decide whether outside connections should consider this neuron as activated or not.
* Thresholding function (step function): (in binary classification) however for multivalve classification, more than 1 neuron output 1, so how to decide which class it should be? This is a problem. 
* Linear function: the derivative of this function is a constant which means if there is an error in prediction then the changes made by back propagation is constant and not depending on the change in input. Another problem, each layer is activated by a linear function then no matter how many layers we have, the final activation function of last layer is nothing but just a linear function of the input of first layer. Lose the ability of stacking layers this way.
* Sigmoid function: non-linear, it gives analog activation unlike step functions. It has smooth gradient too. It is in range(0, 1) instead of (-inf, inf). It wont blow up the activation. however, between x values -2 to 2, Y value is very steep which means any small changes in the values of X in that region will cause values of Y to change significantly. That will have a tendency to bring the Y values to either end of the curve. Vanishing gradients problem because towards the end of the sigmoid function, the Y value tends to respond less to the change of X. 
* Tanh function: it is a scaled sigmoid function. But the gradient is stronger for tanh.
* Relu: non-linear [0, inf] blow up the activation. Sparsity of the activation. Sigmoid and tanh can cause almost all neurons to fire in an analog way, so the activation is dense. ReLu can make it sparse. However, for negative X, the gradient can go towards 0 so those neurons will stop responding to variations in error input. This is called dying ReLu problem. Variation: y=0.01x for x < 0
* Softmax: generalization of the sigmoid function to the case where we want to handle multiple classes. output are in the range(0,1) and sum up to 1 and therefore can be interpreted as probabilities that our input belongs to one of a set of output classes.

##### 6. Back propagation
Use back propagation (training algorithm) to compute the gradient and update the weights. A backpropagation network is a feedforward network trained by backpropagation. Backpropagation is a training algorithm used for a multilayer neural networks. It moves the error information from the end of the network to all the weights inside the network and thus allows for efficient computation of the gradient.<br >
* The backpropagation algorithm can be divided into several steps:
	1. Forward propagation of training data through the network in order to generate output.
	2. Use target value and output value to compute error derivative with respect to output activations.
	3. Backpropagate to compute the derivative of the error with respect to output activations in the previous layer and continue for all hidden layers.
	4. Use the previously calculated derivatives for output and all hidden layers to calculate the error derivative with respect to weights.	
	5. Update the weights.
  
It is usually used to find the gradients of the weights and biases with respect to the cost, but in full generality backpropagation is just an algorithm that efficiently calculates gradients on a computational graph (which is what a neural network is). Thus it can also be used to calculate the gradients of the cost function with respect to the inputs of the neural network.

##### 7. Gradient descent
It is an optimized algorithm used in machine learning to learn values of parameters that minimize the cost function. It is an iterative algorithm, in each iteration, it calculates the gradient of cost function with respect to each parameter and then update the parameter. <br >
* Gradient descent
	1. Stochastic Gradient Descent: Uses only single training example to calculate the gradient and update parameters.
	2. Batch Gradient Descent: Calculate the gradients for the whole dataset and perform just one update at each iteration.
	3. Mini-batch Gradient Descent: Mini-batch gradient is a variation of stochastic gradient descent where instead of single training example, mini-batch of samples is used. It’s one of the most popular optimization algorithms. 

* Advantage of mini-batch gradient descent
	1. Computationally efficient compared to stochastic gradient descent.
	2. Improve generalization by finding flat minima.
	3. Improving convergence, by using mini-batches we approximating the gradient of the entire training set, which might help to avoid local minima.

##### 8. Matrix element-wise multiplication: 相同位置的元素相乘
dot production: a^T*b

##### 9. One hot encoding
encode the categorical features

##### 10. Data normalization
for better convergence during back propagation

##### 11. Weight initialization
close to zero without being too small

##### 12. Hyperparameters
can not learn from the data, need to set before training phrase. Learning rate, number of epochs, batch size

##### 13. CNN
Consists of a set of filters (kernels), slide over the image and compute the dot product between the weights of the filter and the input images.

##### 14. Autoencoder
Learning data codings in a unsupervised manner. The goal is to learn the representation (encoding) for a set of data, typically for dimensionality reduction. Then use the decoder to convert the internal state to the outputs.

##### 15. Prevent overfitting
* dropout
* l1 or l2 regularization
* data augmentation
* early stop

##### 16. Cross entropy
For classification problem

##### 17. Feedforward neural network and recurrent neural network
Feedforward allows signals to travel one way only from input to output. It just compute one fixed-size input to one fixed-size output. The RNN can handle sequential data of arbitrary length.

##### 18. High dimensionality
* delete some dimension: numerical variables, use correlation to find correlated variables
* categorical variables: use chi-square test

##### 19. Gradient
The maximum of derivative is the the direction of the gradient, so we need to update the weight according to the opposite direction of gradient. Just like finding the quickest and steepest way to go down the mountain. 具体化到一元函数中时，梯度方向首先是沿着曲线的切线的，然后取切线向上增长的方向为梯度方向，二元或者多元函数中，梯度向量为函数值f对每个变量的导数，该向量的方向就是梯度的方向. 

##### 20. Batch gradient
All the samples will contribute to update the weight. If the samples are not large, then it is fine it will be very quick to converge. Otherwise, it could take a long time to update. Stochastic gradient descent (SGD): each time, just use one sample to update the weight. So this may not be the global optimum it is just close to the optimum but we can accept that because it is quick. Mini-batch: middle between batch gradient and stochastic gradient descent. So when the training dataset is huge, then SGD can help to reduce the time complexity.

##### 21. PCA
We aim to select fewer components which can explain the maximum variance in the data set. Doing rotation can maximize the difference between variance captured by the component. If not doing this, the effect of PCA will diminish. 

##### 22. Evaluate model
* sensitivity (TP rate)
* specificity (TN rate)
* F measure
* confusion matrix <br >
TP   FN(2) <br >
FP(1)   TN <br >
  
* precision = tp / tp + fp, recall = tp / tp + fn == true positive rate
* ROC: relationship between sensitivity and specificity
* precision-recall (PR) curve might give better representation of the performance

##### 23. Why naive?
Because it assumes that all of the features in a dataset are equally important and independent. These assumption are rarely true in real world.

##### 24. Decision tree
Non-linear interactions

##### 25. Bias
* Errors which can result in failure of entire model and can alter the accuracy
* Low bias: predicted value are close to the actual values, but it loses the generalization capabilities —> can use bagging algorithm (such as random forest) to tackle high variance problems. (voting: classification or averaging: regression). To solve the high variance problem: use regularization techniques

##### 26. Remove correlated variables before PCA. Because it would mislead the PCA to put more emphasis on those related variables.

##### 27. Ensemble learners
They are built on the premise of combining weak uncorrelated models to obtain better predictions, so it might fail when those independent models are performing pretty good. They might just provide the same information.

##### 28. Tf.tensor 
They are objects. (higher dimensional arrays) An object is a symbolic handle to the result of an operation but does not actually hold the values of the operation’s output. Instead, Tensorflow encourages users to build up the complicated expressions as a data flow graph. You then offload the computation of the entire data flow graph to a TensorFlow tf.Session. —> execute more efficiently.

##### 29. feed_dict
It is a dictionary that maps tf.tensor objects to numpy arrays which will be used as the values of those tensors in the execution of a step.

##### 30. tf.stop_gradient
It provides a way to not compute gradient with respect to some variables during back-propagation

##### 31. keep_prob
This value is used to control the dropout rate


##### 32. KNN classification or regression
Find K neighbour around it and then calculate the maximum number of those data points belong to and then classify the data point into that class. It is supervise learning. Lazy learner because it involves minimal training of model and does not use training data to make generalization on unseen dataset.

##### 34. Why add bias
Y = W*x + b: it is linear so without biases, what if all the input x are 0 then no neuron is gonna fired so adding a bias can make them still responsible for the input	
