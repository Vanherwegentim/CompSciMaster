# Artificial Neural Networks and Deep Learning

## List of Questions

### Lecture 2

#### Question 1

**Explain similarities and differences between multilayer perceptron (MLP) and radial basis function (RBF) networks.**

They both are neural networks, the biggest difference between them is the activation function that they use. While the **MLP** primarily uses non-linear activation functions, the RBF network uses radial basis functions as activation functions . These radial basis functions are functions that have a peak at a particular points in space. RBF networks also typically only have 3 layers.

Some radial basis functions include:

- Gaussian
- inverse multiquadratic

![Artificial Neural Network](https://www.saedsayad.com/images/ANN_Gaussian.png)

Some non-linear functions include:

- ReLU (rectified linear unit)
- Sigmoid function

![Sigmoid Function -- from Wolfram MathWorld](https://mathworld.wolfram.com/images/eps-svg/SigmoidFunction_701.svg)

​														*Sigmoid function*

<img src="https://www.nomidl.com/wp-content/uploads/2022/04/image-10.png" alt="What is ReLU and Sigmoid activation function? - Nomidl" style="zoom:50%;" />

​														*ReLU function*

#### Question 2

**What are advantages or disadvantages of multilayer perceptrons versus polynomial expansions?**

While we can use both to approximate functions. The polynomial expansion falls prey to the curse of dimensionality. MLPs are better able to cope with this curse and it was shown that the approximation error becomes independent from the dimension of the input space under certain conditions while this is not the case for polynomial expansions.

Approximation error for

- MLP's with one hidden layer : $O(1/n_h)$
- Polynomial expansions: $O(1/n^{2/n}_p)$



#### Question 3

**Explain the backpropagation algorithm.**

The algorithm is used to train neural nets. It start by initializing all the weights at random. It will then let an input go through the neural network. The output will be compared to what the output is supposed to be and the loss or error will be calculated. The algorithm now will go back from the output layer to the input layer and calculate the error contribution of each neuron and adjust its weight proportionally. This is to be repeated with a lot of training data for the loss to converge.



#### Question 4

**What is the difference between on-line learning and off-line learning with backpropagation?**

**On-line** learning backpropagation: adapt weights each time after presenting a new pattern.

**Off-line** learning backpropagation: adapt weights after presenting all the training patterns to the network.



#### Question 5

**What are the limitations of a perceptron?**

A simple example would be the XOR-gate because to shatter that we would need non-linear decision boundaries which are not possible with just a single perceptron.



#### Question 6

**What does Cover’s theorem tell us about linear separability?**

That if we cannot separate something in the original space linearly that such a solution might be found in a higher dimension.



### Lecture 3

#### Question 1

**Explain the Newton and Levenberg-Marquardt learning of neural networks.**

**Newton's** method is an optimization method that uses the inverse of the hessian matrix and gradient to determine the optimal step. This method converges quadratically which is much faster than the steepest descent algorithm. However this can be computationally expensive as this requires the inversion of the Hessian matrix, which can be large and dense in high-dimensional problems. Another problem is that the hessian matrix often has zero eigenvalues which means that one cannot take the inverse of the matrix. Levenberg-Marquardt and quasi-Newton methods are used to overcome these problems.



The **Levenberg-Marquardt** algorithm is again a method for training neural networks by 

A **damping factor** is also applied to the gradient to balance the trade-off between fast convergence and stability. The algorithm then uses the damped gradient to update the weights of the neural network. The learning rate of this algorithm is not fixed like in previous methods. The learning rate is adjusted **dynamically** based on the **curvature of the error surface**. This allows the algorithm to converge faster while **avoiding overshooting** the optimal solution.

This method combines the benefits of gradient descent and Gauss-Newton methods.

**Hessian matrix:** A matrix of the second-order partial derivatives.

**Jacobian matrix:** A matrix of the first-order partial derivatives.

**TODO**

#### Question 2

**Explain quasi-Newton learning of neural networks.**

Similar to **Newton's** method but with an important difference, that begin that we approximate the **Hessian** matrix instead of computing its exact values. This saves us a lot of time and computation power. The **BFGS algorithm** is an example of a quasi-Newton algorithm. Unfortunately, when the neural network contains many interconnection weights, it becomes hard to store the matrices into computer memory. That's why for large scale neural networks, conjugate gradient methods are to be preferred.



#### Question 3

**Explain conjugate gradient learning of neural networks.**

This kind of learning was developed because the previous algorithms were not feasible on large scale problems.  It reduces the number of iterations required to converge by maintaining a set of **conjugate search directions** that are **orthogonal** to each other and have **no information overlap**. In each iteration the method selects a new direction that is a linear combination of the conjugate search directions and performs a line search to find the minimum along that direction. The step size is chosen such that the new direction remains conjugate to the previous directions.



#### Question 4

**What is the role of a regularization term?**

The regularization term is a penalty term we add to the loss function during training, which penalizes large parameter values or encourages sparsity in the model. This encourages the model to prioritize smaller weights or select only a subset of features. Reducing overfitting and improving generalization.

**Sparsity:** sparsity in a model refers to the property of having a large number of zero or near-zero values in the parameters or features of the model. This allows for more efficient and more interpretable models because only a subset of features are actively contributing to the model's output.

#### Question 5

**What is overfitting? How can this be avoided?**

Overfitting is what happens when we train the neural net too far and instead of being able to generalize well to new, unseen data, it just sort of memorized the data. We can avoid this by using regularization, early stopping and a validation set.



#### Question 6

**What is the effective number of parameters?**

The effective number of parameters refers to the number of non-zero parameters that actively contribute to the model's predictions. In a sparse model, the effective number of parameters is smaller than the total number of parameters in the model because many of the parameters are forced to be zero because the sparsity introduced by regularization.

### Lecture 4

#### Question 1

**What is the difference between least squares (regression) and ridge regression (regularization)? How is this related to the bias-variance trade-off?**

Least squares regression aims to find the line in the case of linear regression or hyperplane in the case of multiple linear regression that minimizes the sum of the squared differences between the observed and predicted values

Ridge regression aims to do the same but it adds a penalty term to the least squares objective function, known as the L2 regularization term. This penalty term helps to shrink the coefficients towards zero, reducing their variance and overfitting.

The bias-variance trade-off is the trade off between bias and variance as we change the penalty term. A large penalty term decreases variance but leads to a larger bias and a small penalty term decreases bias but increases variance. This is why we choose the penalty term so that we minimize both.

**Bias:** the inability for a machine learning method to capture the true relationship. The higher the bias the worse the model is able to predict.

**Variance:** The difference between the performance on the training set and the performance on the test is what we call the variance. If the variance is high, it probably means that the model is overfitting the training set and thus does bad on the test set.

#### Question 2

**Explain cross-validation.**

Cross validation is a method used in training neural nets, we make the neural net split up our train data in a train and validation set with different proportions. For example in 10fold cross validation we split the training set in 10 parts, taking 9 for training and 1 for validation and then alternate the assignments. Doing this we can prevent the model from overfitting.



#### Question 3

**Explain complexity criteria.**

Complexity criteria states that one should not only try to minimize training errors but also keep the model complexity as low as possible. (In essence Occam's razor).



#### Question 4

**Discuss pruning algorithms.**

In order to improve the generalization performance of the trained models one can remove interconnection weights that are irrelevant. This is what we call pruning.

**Optimal brain damage** works as follows

1. Train your network to a minimum of the error function

2. Compute the saliency values for all the interconnection weights
3. Sort the weights by saliency and delete the low-saliency weights
4. Go to step 1 and repeat until some stopping criterion is reached



**Optimal brain surgeon** works as follows

1. Train your network to a minimum of the error function
2. Calculate which weights can be removed without introducing a lot of error using the inverse Hessian matrix
3. Adjust remaining weights to account for removal of the weights.
4. Go to 2 and repeat until some stopping criterion is reached



**Weight elimination**: this algorithm is more likely to eliminate weights than the weight decay method.

**TODO**

 OBS has better performance than OBD because OBD completely removes the connection with low-saliency from the network while , removing the influence of the weight. This helps to maintain structural integrity.

**saliency values:** The values that describe the relative importance of interconnection weights.



#### Question 5

**Explain the committee networks method.**

This method is instead of taking one model and training, we take several and combine them to make a sort of committee of networks. This committee of networks can outperform the best single network. This comes with a few disadvantages because we need to train a lot more.





### Lecture 5

#### Question 1

**Explain the Occam’s razor principle.**

Occam's razors says that the simplest explanation should be the one most preferred. For us, this would mean that we prefer the simplest model. Simple models tend to make precise predictions. Complex models by their nature , are capable of making a greater variety of predictions



#### Question 2

**What is the difference between parameters and hyperparameters when training multilayer perceptrons?**

Parameters are the variables of model that are learned from the training data. The values that the model adjusts during the training to minimize the loss function. These include weights and biases associated with the connections between neurons in the network.

Hyperparameters are the configurations of the neural network that are set before the training beings. These are not learned but set by the developer. Hyperparameter include the learning rate, number of hidden layers, number of neurons in each layer, etc...

#### Question 3

**What is the role of the prior distribution in Bayesian learning of neural networks**

The prior distribution represents prior beliefs about the model's parameters. It incorporates prior knowledge and influences the posterior distribution which are the update beliefs after observing the data.

#### Question 4

**What is the difference between the number of parameters and the effective number of parameters?**

The number of parameters refers to the total count of weights in a model while the effective number of parameters refers to the numbers of parameters that aren't close to zero so they have an actually impact on the output. The amount parameters and effective number of parameters might not match because regularization techniques and vanishing gradients.

#### Question 5

**How does one characterize uncertainties on predictions in a Bayesian learning framework?**

TODO

### Lecture 6

#### Question 1

**What is the working principle of associative memories from a dynamical systems point of view?**

Associative memories work by learning the relationships between the patterns to be stored. It does this by changing the weights between connections so that when it is in such a pattern, the energy function is at its lowest. When a partial or noisy version of the pattern is introduced, it can retrieve the stored pattern by self-organizing to the closest learned pattern.



#### Question 2

**What is the Hebb rule for storing patterns in associative memories and why does it work?**

The **Hebbian** learning rule is based on the idea that "neurons that fire together, wire together". The rule states that the weight between two neurons should be increased if both neurons are active at the same time and deceased if one neuron is active while the other is not.



#### Question 3

**What determines the storage capacity in associative memories?**

The storage capacity is limited by the ability to retrieve stored information without errors and the potential interference between stored patterns.



#### Question 4

**When solving the TSP problem using a Hopfield network, how are cities and a tour being represented?**

To solve TSP using a Hopfield network, we first need to define the energy function. In the case of TSP, the energy function is defined as the sum of the distances between adjacent cities along the route, where the route must visit every city exactly once







### Lecture 8 

#### Question 1

**How can a multilayer perceptron be used for time-series prediction?** 

We can create a feedforward neural net that takes the known values for x amount of time-values as input (with x being the amount of lagg we decide to use). Using this input we can let the model predict the most probable value for the next timestamp. Afterwards we can add this prediction to our input in a sliding window fashion to then predict the next timestamp, and so on. 

#### Question 2

**How can one use neural networks in different model structures for system identification?** 

We can use neural networks in system identifications like time series by training on the input and output data and letting the neural net find out how the output is correlated to the input data. Doing this should allow one to discovering the working of the system and predict outputs.

#### Question 3

**Explain the use of dynamic backpropagation.** 

 

#### Question 4

**Explain the use of neural networks for control applications.** 

Neural networks can be used in control applications by first observing an expert (training data) and training a model to mimic the actions of that expert (output data). The model can be trained by using reinforcement learning and giving the model a reward when performing the same action as the expert.



### Lecture 9 

#### Question 1

**What are advantages of support vector machines in comparison with classical multilayer perceptrons?** 

SVM’s are more interpretable than classical multiplayer perceptron’s. they also maximize the distance between classes. SVM’s are also more computational efficient, and less effected by outliers. 

#### Question 2

**What is the kernel trick in support vector machines?** 

The kernel trick is used to transform data that isn’t linearly separable to another dimension so we can separate it without actually calculating the features vectors in that dimension, thus making it more computational efficient.

#### Question 3

**What is a support vector?** 

A support vector is a line we draw to separate data. SVM’s try to maximize the distance between 2 differently classified points to get more accurate predictions. These vectors then get stored and used to make predictions about the test data. By using support vectors instead of the datapoints we don’t have to store all the training data but only the support vectors.

####  Question 4

**What is a primal and a dual problem in support vector machines?** 

Bullshit negro



### Lecture 10 

#### Question 1

**• Give motivations for considering the use of more hidden layers in multilayer feedforward neural networks.** 

Using more hidden layers can give our model more possibilities to combine features learned in the early layers. Generally, this results in models that can work out more complex tasks.

#### Question 2

**Explain the pre-training and fine-tuning steps when combining an autoencoder with a classifier.**

During the pre-training we train autoencoders to condense the data/learn important features, this is done unsupervised. When starting to fine-tune we append a classifier and add the training data assignments to use the features learned from the autoencoders to predict the classification of test data.

#### Question 3

**• What are possible difficulties for training deep networks?** 

There are numerous difficulties in training deep networks. Some of the most known ones which we also saw while working on our report are overfitting, vanishing/exploding gradients, and computational difficulties. Further they also get less interpretable and explainable making them more and more of a black box model.

#### Question 4

**Explain stacked autoencoders.** 

A regular autoencoder is meant to reduce the dimensionality of input data, stacked autoencoders go further on this by condensing the condensed data even further. For example, a regular autoencoder can learn to represent digits by their binary encoding, stacked autoencoders can take this 1 step r resulting in smaller more meaningful features.

#### Question 5

**Explain convolution and max-pooling in convolutional neural networks.** 

Convolutional layers work by taking the average/median of multiple values (in a box) to condense the data. Max-pooling layers kind off work the same as convolutional layers as they condense the data, but instead of taking the average/mean they take the highest occurring value in the set of cells.



### Lecture 11

#### Question 1

 **Explain Restricted Boltzmann Machines.** 

A restricted Boltzmann machine is a kind of neural network with 2 fully connected layers. It is meant to generate or reduce dimensionality in data. 

#### Question 2

**Explain Deep Boltzmann Machines.** 

Deep Boltzmann Machines is a variant of restricted Boltzmann machines but with more fully connected layers. Using more layers allows the model to learn more complex features and generate more accurate data. Because of the increased layers the computational complexity also increases

#### Question 3

**Discuss Wasserstein training of Restricted Boltzmann Machines.** 

Wasserstein distance in RBMs aims to reduce the distance between generated data and input data as much as possible. By using the Wasserstein distance we get less noise in the generated data which In turns makes the data more accurate. It is easily visualized by using it on image generations for example the digits dataset we used in our reports. Here we saw that the generated data with Wasserstein had less random white pixels.

#### Question 4

**Discuss Generative Adversarial Networks.**

Generative adversial networks can be described as 2 separate neural nets. A generator which tries to generate “fake” input data and a discriminator which tries to distinguish fake and real input data. If a discriminator can do this it means our generator hasn’t done a good job and needs to adjust its weights.

