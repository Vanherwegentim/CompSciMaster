# Artificial Neural Networks and Deep Learning

## List of Questions

### Lecture 2

#### Question 1

**Explain similarities and differences between multilayer perceptron (MLP) and radial basis function (RBF) networks.**

They both are neural networks, the biggest difference between them is the activation function that they use. While the **MLP** primarily uses non-linear activation functions, the RBF network uses radial basis functions as activation functions . These radial basis functions are functions that have a peak at a particular points in space. RBF networks also typically only have 3 layers.

Some radial basis functions include:

- Gaussian
- inverse multiquadratic

Some non-linear functions include:

- ReLU (rectified linear unit)
- Sigmoid function

![Sigmoid Function -- from Wolfram MathWorld](https://mathworld.wolfram.com/images/eps-svg/SigmoidFunction_701.svg)

​														*Sigmoid function*

<img src="https://www.nomidl.com/wp-content/uploads/2022/04/image-10.png" alt="What is ReLU and Sigmoid activation function? - Nomidl" style="zoom:50%;" />

​														*ReLU function*

#### Question 2

**What are advantages or disadvantages of multilayer perceptrons versus polynomial expansions?**

Multilayer perceptrons are artificial neural networks used for function approximation using nonlinear activation functions. Polynomial expansions are also used for function approximation but it is done using a set of basis functions. These can be more interpretable and can provide insight into the underlying structure of the function being approximated while MLP is more flexible and can learn any nonlinear function. TODO?



#### Question 3

**Explain the backpropagation algorithm.**

The algorithm is used to train neural nets. It start by initializing all the weights at random. It will then let an input go through the neural network. The output will be compared to what the output is supposed to be and the loss or error will be calculated. The algorithm now will go back from the output layer to the input layer and calculate the error contribution of each neuron and adjust its weight proportionally. This is to be repeated with a lot of training data for the loss to converge.



#### Question 4

**What is the difference between on-line learning and off-line learning with backpropagation?**

**On-line** learning backpropagation: adapt weights each time after presenting a new pattern.

**Off-line** learning backpropagation: adapt weights after presenting all the training patters to the network.



#### Question 5

**What are the limitations of a perceptron?**

A simple example would be the XOR-gate because to shatter that we would need non-linear decision boundaries which are not possible with just a single perceptron.



#### Question 6

**What does Cover’s theorem tell us about linear separability?**

That if we cannot separate something in the original space linearly that such a solution might be found in a higher dimension.



### Lecture 3

#### Question 1

**Explain the Newton and Levenberg-Marquardt learning of neural networks.**

**Newton's** method is a powerful optimization method that uses second-order information about the cost function to find the minimum more quickly. It uses the cost function locally as a quadratic function and uses this to compute the direction and step-size for the weights updates. However this can be computationally expensive as this requires the inversion of the Hessian matrix, which can be large and dense in high-dimensional problems.



The **Levenberg-Marquardt** algorithm is again a method for training neural networks by minimizing the difference between the predicted outputs and the actual outputs of the training data. The algorithm starts by computing the **Jacobian** matrix, which is a **matrix of partial derivatives** of the outputs. The matrix is used to calculate the gradient of the error which gives the direction of steepest descent for the error.

A **damping factor** is also applied to the gradient to balance the trade-off between fast convergence and stability. The algorithm then uses the damped gradient to update the weights of the neural network. The learning rate of this algorithm is not fixed like in previous methods. The learning rate is adjusted **dynamically** based on the **curvature of the error surface**. This allows the algorithm to converge faster while **avoiding overshooting** the optimal solution.

This method combines the benefits of gradient descent and Gauss-Newton methods.



#### Question 2

**Explain quasi-Newton learning of neural networks.**

Similar to **Newton's** method but with an important difference, that begin that we approximate the **Hessian** matrix instead of computing its exact values. This saves us a lot of time and computation power. The **BFGS algorithm** is an example of a quasi-Newton algorithm.



#### Question 3

**Explain conjugate gradient learning of neural networks.**

This kind of learning was developed because the previous algorithms were not feasible on large scale problems.  It reduces the number of iterations required to converge by maintaining a set of **conjugate search directions** that are **orthogonal** to each other and have **no information overlap**. In each iteration the method selects a new direction that is a linear combination of the conjugate search directions and performs a line search to find the minimum along that direction. The step size is chosen such that the new direction remains conjugate to the previous directions.



#### Question 4

**What is the role of a regularization term?**

The regularization term is a penalty term we add to the loss function during training to prevent overfitting and encourage better generalizing.



#### Question 5

**What is overfitting? How can this be avoided?**

Overfitting is what happens when we train the neural net too far and instead of being able to generalize well to new, unseen data, it just sort of learned the data by heart. We can avoid this by using regularization, early stopping and a validation set.



#### Question 6

**What is the effective number of parameters?**

The effective number of parameters is a way to quantify the complexity of a model while taking into account the impact of regularization. Essentially, it measures the number of parameters that the model is actually able to learn from the data, rather than the total number of parameters in the model.



### Lecture 4

#### Question 1

**What is the difference between least squares and ridge regression? How is this related to the bias-variance trade-off?**

TODO





#### Question 2

**Explain cross-validation.**

Cross-validation is variant of validation meaning that we will have 2 different data sets, a data set for training and one for testing. The change that cross-validation makes is that there are several different training and testing data sets. The model is then evaluated on this and the results are averaged to get a more accurate estimate of the model's performance



#### Question 3

**Explain complexity criteria.**

Complexity criteria refers to the methods used to determine the optimal level of complexity or size for your model to balance overfitting and generalization. TODO?



#### Question 4

**Discuss pruning algorithms.**

The pruning algorithms are Optimal Brain Damage and Optimal Brain Surgeon(I think, maybe weight elimination also?). OBS has better performance than OBD. OBD completely removes the neuron from the network while OBS just sets the weight to 0, removing the influence of the neuron. This helps to maintain structural integrity.



#### Question 5

**Explain the committee networks method.**

This method is instead of taking one model and training, we take several and combine them to make a sort of committee of networks. This committee of networks can outperform the best single network. This comes with a few disadvantages because we need to train a lot more.





### Lecture 5

#### Question 1

**Explain the Occam’s razor principle.**

Occam's razors says that the simplest explanation should be the one most preferred. For us, this would mean that we prefer the simplest model. TODO



#### Question 2

**What is the difference between parameters and hyperparameters when training multilayer perceptrons?**

TODO

#### Question 3

**What is the role of the prior distribution in Bayesian learning of neural networks**



#### Question 4

**What is the difference between the number of parameters and the effective number of parameters?**



#### Question 5

**How does one characterize uncertainties on predictions in a Bayesian learning framework?**



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



### Lecture 7

#### Question 1

How can one do dimensionality reduction using linear principal component analysis and nonlinear principal component analysis

