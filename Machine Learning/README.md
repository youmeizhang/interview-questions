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
Y = W\*x + b: it is linear so without biases, what if all the input x are 0 then no neuron is gonna fired so adding a bias can make them still responsible for the input	

##### 35. Logistic regression
it is a technique to output a binary outcome from a linear combination of predictor variables. For example, if you want to predict whether a political leader will win the election or not, then the prediction is binary (0/1) (lose/win). The predictor variables here would be the amount of money spent for election campaigning of a particular candidate, the amount of time spent in campaigning, etc.

##### 36. Normal distribution
data is usually distributed in different ways with a bias to the left or to the right or it can all be jumbled up. However, there are chances that data is distributed around a central value without any bias to the left or right and reaches normal distribution in the form of a bell shaped curve. The random variables are distributed in the form of an symmetrical bell shaped curve.

##### 37. Linear regression
it is used to find the linear relationship between target and one or more predictors. There are two types: simple and multiple. simple: find the relationship between two continuous variables, one is predictor and the other is response.

##### 38. Interpolation and Extrapolation
Estimating a value from 2 known values from a list of values is Interpolation. Extrapolation is approximating a value by extending a known set of values or facts.

##### 39. K-means clustering
K-means clustering is a type of unsupervised learning, which is used when you have unlabeled data (i.e., data without defined categories or groups). The goal of this algorithm is to find groups in the data, with the number of groups represented by the variable K. The algorithm works iteratively to assign each data point to one of K groups based on the features that are provided. Data points are clustered based on feature similarity

##### 40. Collaborative filtering
The process of filtering used by most of the recommender systems to find patterns or information by collaborating viewpoints, various data sources and multiple agents.

##### 41. Are expected value and mean value different?
* mean: the arithmetic mean which means the sum divided by the total number of observations. 
* expectation: the connotation is that this is a theoretical mean. For example flipping a coin 10 times, 1 for heads and 0 for tails, then you get 6 for heads and 4 for tails, then mean is (1*6 + 0*4) / 10 = 0.6 but expectation is (0.5*0 + 0.5*1)=0.5

##### 42. P-value 
it is used to determine the significance of results after a hypothesis test in statistics. It helps reader to draw conclusions and is always between 0 and 1<br >
* P-value > 0.05 the null hypothesis can not be rejected
* P-value <= 0.05 the null hypothesis can be rejected
* P-value = 0.05 it is possible to go either way

##### 43. Recommender
SVD is a matrix factorization technique that use to reduce space dimension of n to k where k < n. It can be thought of finding 2 matrix whose product is the original matrix. (user-item rating matrix). Each item can be represented by a vector `qi`. Similarly each user can be represented by a vector `pu` such that the dot product of those 2 vectors is the expected rating. `qi` and `pu` can be found in such a way that the square error difference between their dot product and the known rating in the user-item matrix is minimum. Minimize (p,q) sum (r-q^Tp)^2

* Surprise: to implement SVD (singular value decomposition)
* Spark mllib: to fill the missing entries of a user-item association matrix. In spark mllib als algorithm to learn the latent factors
* ALS: it is also a matrix factorization algorithm
* Movielen dataset: 27k movies, 138k users


##### 44. Sentence similarity
remove the stops words and then run word2vec on the words in both sentences and sum up the vectors in one sentence, and sum up the vectors in the other sentence and then find the difference between the sums. In this way, we don’t need to worry about the order in the sentence. But it is not a very good solution.

##### 45. Word2vec
It encodes the semantic meaning of words into dense vectors. It contains two distinct models: CBOW and Skip-gram with or without negative sampling, they are both shallow neural models.
* main idea: distributional hypothesis similar words appear in similar contexts of words around them. Word2vec model using a single hidden layer and fully connected neural network. The neutrons in the hidden layer are all linear neutrons. The input layer has same number neutrons as the words in the vocabulary for training. The hidden layer size is set to the dimensionality of the resulting word vectors. The size of the output is the same the input layer.
* cbow:  given several words, learn to predict the word by the context, or maximize the probability of the target word by looking at the context
* skip-gram: given a word, learn to predict the context 

Word2vec has a large amount of corpus and it aims at news field. So if you are training a model that generating poem then word2vec might not be a good choice for you. To train your own word2vec model, we can import the word2vec model trained by google first and then feed your own dataset into it. The model is actually a three layer models. This really require a large amount of data to make it more accurate because it encodes the semantic information.

##### 46. How to compute TF-IDF matrix:
List all the words appear in all of the documents, and then the first column is the name of documents. TF: For each word in each document, calculate the occurrence in that document / total number of words in that document. IDF: log(total number of documents / the number documents that contain that word), then TF\*IDF. Then for each document we can view it as a vector with these values so that we can calculate the cosine similarity or other kind of similarity methods.cosine similarity =  Ab / |a| |b|

##### 47. SVM
SVM is a discriminative classifier formally defined by a separating hyperplane. Given labeled training data, the algorithm outputs an optimal hyperplane which categorizes new examples. In 2D, the hyperplane is a line dividing a plane in two parts where in each class lay in either side. If it is not linear in a certain dimension, we can use kernel. (Such as a circle) there is no line that can separate the two classes in x-y plane so we apply transformation and add one more dimension as we call it z-axis. Let us assume value of points on z plane, w = x^2 + y^2. Then we have a line. And we transform back the data

##### 48. Boosting
It is about building a stronger classifier based on some weak classifiers. Which means we have some weak classifiers and then next time, we just increase the weights of those classifiers with high accuracy and decrease the weights of those classifiers with low accuracy. (adjust the weights based on the error of previous model)

##### 49. Gradient boosting
It is one kind of boosting, but each time it builds a new model based on the gradient decreasing direction of loss function of the previous model. Because if the loss function is large, then this model has low accuracy and then we build model that can decrease this loss function then we are improving the model. GBDT is one of the classical method of gradient boosting. The difference between GBDT and original boosting is calculating the residual.  Since the original boosting use square loss function, it can calculate the residual directly. But Gradient Boosting use general loss function, so it need to use negative gradient for approximating the residual.

#### 50. GBDT
* Based on residual
* Based on gradient

But there are basically all based on the gradient

#### 51. Residual
The difference between the predicted value and real value. Above the regression line, it is positive and below the regression line, it is negative. This is only true when the it is the square loss function. If not, we can use negative gradient to approximate it.

#### 52. Linear regression and svm
* SVM: find a hyperplane in an n-dimensional space (n is the number of features) that has the maximum margin to distinct the data points. 2 features, a line. 3 features, a 2-dimensional plane.
* Logistic Regression: we take the output of a linear function and then squash it into [0,1] with sigmoid function. Then if the value larger than a threshold, we assign it label 1 otherwise, 0. It is sensitive to outliers. Logistic loss doesn’t go to zero. Difference between this and linear regression: output continues outcome
	
* Difference: logistic regression produces probability while svm produces 1 or 0. The choice of hyperplane is not influenced by support vector. Why it is not popular anymore? 

#### 53. Why svm is not popular anymore?
* Now data is in different kinds of forms, images, videos, text and so on, they are hardly given as feature-vectors. So you need to do feature extraction which requires lots of expertise in that domains.
* Xgboost can run faster and beat it
* If data is not linear, you need to have kernel to model it
* SVM is originally for binary classification even though you can do the 1-against-all

#### 54. Calculate the sentiment score for each sentence
For each word, we can refer to the predefined word list such as ANEW which contain the manual score for each word. For a sentence, we might sum up the scores of words inside the sentence, or take the maximum one or by learning using neural network but we need to have the training data such as yelp.

#### 55. Problem of TF-IDF 
无法捕捉上下文之间的联系

#### 56. Logistic regression can deal with non-linear problem but need to use kernel trick

#### 57. Different loss function
* 0-1损失函数
* 平方损失函数(square error loss)
* 交叉熵损失函数(cross-entropy)
* Hinge损失函数
	
#### 58. Adaboost 通过提升错分数据点的权重来定位模型的不足而gradient boosting通过计算gradient。 Adaboost adaptive原因是基于前一个模型进行调整

#### 59. PCA and SVD
Both are dimension reduction methods
* PCA: choose the direction with more information for adjusting, choose the feature vector with maximum feature value
* SVD: 

#### 60. LSTM中每个gate的作用是什么，为什么跟RNN比起来，LSTM可以防止梯度消失
* input gate
* output gate
* forget gate
	
#### 61. GRU
* update gate: helps the model to determine how much the past information needs to be passed along to the future
* reset gate: from the model to decide how much of the past information to forget

#### 62. Pooling
* reduce features
* reduce parameters
* keep invariant

#### 63. Why gradient vanishing?
due to the usage of inappropriate activation function, such as sigmoid. 
solution: use relu and other kinds of activation function, use batch normalization (help to speed up the training, solve the problem of gradient vanishing and gradient exploding), resnet

#### 64. Gradient exploding
the initial weights are initialized quite large
solution: a. Use gradient clip, weights regularization 

#### 65. 神经网络不收敛的11个常见问题
没有对数据进行归一化, 忘记检查输入和输出, 没有对数据进行预处理, 没有对数据正则化, 使用过大的样本, 使用不正确的学习率, 在输出层使用错误的激活函数, 网络中包含坏梯度, 初始化权重错误, 过深的网络, 隐藏单元数量错误

#### 66. Reinforce learning
No label but something similar to label: reward, can decide if this decision is good or bad, if good, it knows then next time it will do the same thing again. Continuous trying and making mistakes (iterating). One action executed in a certain sate and then get reward, so that the model know which action is optimal
	
#### 67. Transfer learning
* Feature extractor
* Fine tune the convNet, not only replace and retrain the classifier but also fine tune some of the layers
* Pretrained model
* 两个domain必须是相关的

#### 68. 生成对抗网络(GAN)
可以帮忙生成数据，提高数据数量。generator, discriminator: to tell if an image is real or generated. For any input, discriminator output 0 - 1 to indicate the probability of this image being real. Optimization process: g fixed, optimize d, then d fixed, optimize g, then continue until converge. It is very sensitive to hyper parameters need to design it very carefully. If the discriminator is trained very well then the generated model will get very worse because it use jensen-shannon distance. However if two distributed sample spaces do not overlap with each other, this distance can not be defined. Use Wasserstein distance (WGAN)

#### 69. Why Svm is better than other
good generalization ability due to its algorithm. It uses margin, so this is the description of the data distribution. 得到对数据分布的结构化描述，因此减低了对数据规模和数据分布的要求. 它是一个凸优化问题，因此局部最优解一定是全局最优解的优点

#### 70. Classification
* binary classification problem, we can use logistic regression. Logistic function: sigmoid function S-shaped curve that take any real-valued number and map it into a value between 0 and 1.

* Cross Entropy measures the performance of a classification model where the prediction input is a probability value between 0 and 1

#### 71. Why entropy is a better way to measure the performance of a model?
It can reflect the uncertainty. 

#### 72. Loss function
* Cross Entropy: it increases as the predicted probability diverges from the actual label. (cross-entropy and log loss are slightly different depending on context, but in ML when calculating error rates between 0 and 1 they resolve to the same thing)
* MSE: mean squared error L2 loss sum(square of subtraction value between real value and predicted value) / size
* Hinge
* Huber
* KL-Divergence
* MAE: mean absolute error or L1 loss

#### 73. Accuracy
the count of predictions where predicted value equals to the actual value
log loss: takes into account of uncertainty of your prediction based on how much it varies fro the actual label

#### 74. Text similarity
* jaccard similarity: size of intersection divided by size of union of two sets (no need to convert into vector)
* cosine similarity: cosine of angle between two vectors, so need to convert sentences into vectors, one way to do that: bag of words with TF or TF-IDF
** calculate the term frequency using bag of words
** normalize the term frequency with the respective magnitudes or L2 norms
** dot product

#### 75. How word2vec works

#### 76. Feedforward propagation and back forward propagation

#### 77. Validation set and test set difference

#### 78. How do you tune the model

#### 79. Restful 

#### 80. How would you improve the throughput of a model? For example, so many photos at the same time in a website

#### 81. Forward propagation
We start from the input we have, we pass them through the network layer and calculate the actual output of the model straight forwardly. Because the calculation flow is going in the natural forward direction from the input through the neural network to the output, it is called forward propagation

#### 82. Backpropagation
It is to update each of the weights in the network so that they cause the output to be closer to the target output, so it can minimize the error for each output neuron and the network. It calculates the gradient and then send the gradient to the optimizer so as to update the weights. Backpropagation is usually used to find the gradients of the weights and biases with respect to the cost.

#### 83. Transfer learning
* ConvNet as fixed feature extractor: remove FC layer 
* Fine-tuning the ConvNet: replace and retrain the classifier on top of ConvNet on the new dataset, fine-tune the weights of the pertained network by continuing the backpropagation. Maybe fine-tune the last few layers specifically
* Pretrained models: since it takes a long time to train


