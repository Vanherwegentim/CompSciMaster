## General concepts

1. What is the difference between inductive inference and predictive learning?
   - 
2. How does semi-supervised learning differ from supervised and unsupervised learning?
   - 
3. Can you provide an example of a regression task?
   - 
4. What is the purpose of clustering in machine learning?
   - 
5. How does Q-learning differ from other forms of reinforcement learning?
   - 



## Trees

1. How does a Classification tree differ from a Regression tree in decision trees?
   - 
2. What does CART stand for and what does it refer to in the context of decision trees?
   - 
3. What are the main reasons to use decision trees in machine learning?
   -  
   - 
   - 
4. What are the two important details that need to be filled in for decision trees?
   - 
   - 
5. How does oblique trees differ from normal trees in decision trees?



## ANN

1. What is the difference between a perceptron and multi-layer perceptrons?
   - A perceptron is a single neuron taking some input and outputting a certain value given the threshhold
   - Multi layer perceptrons is already a network of several perceptrons with an input layer, hidden layer and output layer. This multilayer perceptron can already approximate most non-linear functions and boolean ...
2. How does a sigmoid function differ from a threshold function in a neuron?
   - A sigmoid function is different from the usual threshhold function because the change is gradual which has the advantage of being more robust to noise
3. Can you explain what a convolution layer in a CNN does
   - extracting features from input data like edges, textures and patterns..
4. What is the role of pooling layers in a CNN?
   - The pooling layers reduces complexity and reduce the granularity of the signal
5. How does training an ANN differ from training a decision tree or linear regression?
   - The training in ANN takes very long and requires a lot more data than decision trees or linear regression
     It is is done as follow :
     1. Take a new instance (x,y)
     2. Input the x and then compare f(x) to y
     3. Change the paramters in the ANN so f(x) comes closer to y

## Support vector machines and naive bayes

1. What is the goal of Support Vector Machines?
   1. Support vector machines are a type of machine learning that is used for classification. It tries to find the hyperplane that perfectly separates the given classes. 
2. How does a kernel function help in the process of linear separation?
   1. The kernel function transforms the data to a higher dimension where a hyplerplane that separates the data might exist
3. What is Bayes' rule?
   1. P(A|B) = P(B|A)P(A)/P(B)
4. What is the difference between the MAP model and Maximum Likelihood (ML) model?
   1. MAP model uses a prior in the formula
      $h_map = P(D|h)P(h)$
   2. $h_ML = P(D|h)$
5. What is the principle of Occam's razor?
   1. The simplest hypothesis is usually the best
6. How is Bayes' rule used for classification in Naive Bayes?
   1. argmax P(c|x_1,x_n) = argmax P(x_1...x_n|c)
7. ...
8. What is the problem with attribute value never being observed for a class?
   1. Say we like to make predictions about a duck population but we haven't observed a black duck. Given our data, if we would compute the probabilities we would get 0 but we know this wouldn't be corrrect because black ducks do exist.
9. How does m-estimate help with this problem?
   1. $P(x|c) = \#(xi,c)+mq/(\#c+m)$
10. What is the difference between generative and discriminative models?
    1. Generative models define P(A,B)
    2. discriminative models define P(A|B)
11. How is Naive Bayes classified as a generative model?
    1. Naive bayes uses P(A,B)



## Ensembles

1. What is the main idea behind ensembles in machine learning?
   1. What if we combine several classifiers, could we get something that is better than either classifier
2. How does voting work in ensemble learning?
   1. We let several learners work in parallel to create a hypothesis and then choose the hypothesis the most learners returned
3. Can you explain the process of Bagging in ensemble learning?
   1. Bagging is using differen subsets of the data to train the leaners. This can improve the performance because the learners are less likely make the same mistakes
4. How is Boosting different from Bagging in ensemble learning?
   1. Boosting is using a weighted data set to train a learner. When give more weights to the predictions that were incorrect and feed again into a learner. So new hypotheses will focus on these examples. ADABOOST
5. Can you explain the process of Stacking in ensemble learning?
   1. Stacking is a method for combing multiple models to improve perforamce. It involves training several models and then using these models to train a meta model that learns how to best combine the outputs of the base models to make pr
6. How does Planning differ from Behavioral Cloning and Reinforcement Learning in terms of learning a policy?
   1. We build a plan that reaches the desired state
7. Can you explain the concept of behavioral cloning in reinforcement learning?
8. How does the principle of reinforcement learning differ from other forms of learning?
9. What is the complete algorithm for Policy Iteration in reinforcement learning?
10. What is the generalization process in reinforcement learning, specifically in Q-learning?



1. What is the difference between association rules and classification rules?
2. What is the APRIORI algorithm and how is it used in association rules?
3. How does the k-NN algorithm classify a new instance?
4. What is the curse of dimensionality and how does it affect k-NN?
5. What is a ROC analysis and how is it used in evaluating a classifier?
6. How does sample complexity differ when the learner proposes instances versus the teacher providing both instances and labels?
7. What is meant by "learner is PAC-learnable"?
8. How is ROC analysis used in case of numerical values instead of nominal values?
9. What is the difference between correlation and accuracy in evaluating a classifier?
10. How is the sample complexity related to the VC-dimension of a hypothesis space?