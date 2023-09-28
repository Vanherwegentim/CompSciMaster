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
   1. Stacking is a method for combing multiple models to improve perforamce. It involves training several models and then using these models to train a meta model that learns how to best combine the outputs of the base models to make predictions
6. How does Planning differ from Behavioral Cloning and Reinforcement Learning in terms of learning a policy?
   1. We build a plan that reaches the desired state, unfortunately this requires a full model of the environment
7. Can you explain the concept of behavioral cloning in reinforcement learning?
   1. The agent learns how to traverse the environment from a teacher, unforunately this means that the agent can barely do better than the teacher and is prone to overfitting
8. How does the principle of reinforcement learning differ from other forms of learning?
   1. We let the agent run free in the environment but reward good behavour and punish bad
9. What is the complete algorithm for Policy Iteration in reinforcement learning?
   1. Policy evaluation -> policy improvement -> policy evaluation
10. What is the generalization process in reinforcement learning, specifically in Q-learning?
    1. Generalization in reinforcement learning is a problem. In Q-learning we can have millions of state-action pairs with some Q-values. This is too much data. So what we do we create a lot less and then feed these pairs with Q-values into a learner that creates a model that can predict the Q-value for a a certain input.

​		

1. What is the difference between association rules and classification rules?

   1. Assocation rules are meant to describe the underlying patters or relationship. Classification rules are meant predict an output given some input

2. What is the APRIORI algorithm and how is it used in association rules?

   it is an example

3. How does the k-NN algorithm classify a new instance?

   1. the k-nn algorithms uses the similarty of data points. Given a new data points it will find the k nearest data points and then classify it depending on those.

4. What is the curse of dimensionality and how does it affect k-NN?

   1. The curse of dimensionality is that its very hard to classify things in higher dimension because they are not intuitive and data points could spread very far from each other. k-NN does not perform well in high dimensions

5. What is a ROC analysis and how is it used in evaluating a classifier?

   1. ROC analysis is the comparing of different classifiers based some threshold and the TP and FP

6. What does PAC stand for?

   1. probably approximately correct

1. What is the definition of shattering and what does it tell us about a hypothesis space?
   1. A hypothesis space H shatters a set of instances T, if for whatever labeling, there exists a hypothesis in H a that shatters that labeling.
2. What is the definition of VC-dimension and how does it relate to a hypothesis space?
   1. The VC-dimension is the cardinality of the largest set of points that is shattered by H
3. What is the upper bound for VC-dimension in terms of the size of a hypothesis space?
   1. Upper bound vc dimension $VC(H) \le log_2(|H|)$ and if a hypothesis H shatters a T with n points then that H contains at least $2^n$ hypotheses





1. What is the purpose of the LearnOneRule algorithm?
   1. learn a ruleset from a dataset to be able to make predictions about new data
2. How does the top-down variant of the LearnOneRule algorithm work?
   1. The top-down variant grabs the most general instance in the training data and uses this to start from, it then goes over all the other instance in the set and improves the accuracy while keeping the same coverage
3. How does the bottom-up variant of the LearnOneRule algorithm differ from the top-down variant?
   1. Instead of starting with the most general rule, it start with the most specialized rule and then goes over all the instances while holding the same accuracy and increasing coverage.
4. What is RIPPER?
   1. Currently of the best ruleset learners
5. What is the m-estimate of a rule and how is it calculated?
   1. The accuracy rule that predicts class C is p/(p+n) with p = #positives and n#negatives
   2. The m-estimate of a rule is $p+mq/(p + n + m)$