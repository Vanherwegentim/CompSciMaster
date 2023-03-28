# Zelftest 2

What is the role of pooling layers in a CNN?

- reduce dimensionality and reduce granularity

What is the difference between a k-input perceptron and multi-layer perceptrons?

- a k-input perceptron is a single neuron with k-input. it's a linear separator. If we the input is not linearely separable, the perceptron will not find a hypothesis
- a multi perceptron is already a simple ann. It is a universal approximator, any function can be approximated

How does m-estimate help with this problem?

- P(x|i) = #(x,i)+mq/(#c+m)

What does PAC stand for?

- probably approximately correct, We take random sample of the population and feed this to the learner. It is possible the hypothesis is not correct because the sample was not representative. That's why we "relax" the learner, it can be probably approximately correct

What is the definition of shattering and what does it tell us about a hypothesis space?

- A hypothesis space H shatters an set of instances T, if for each positive and negative labeling of the data points there exists a hypothesis h in H that is consist with those data points.
- If H shatters n data points then H contains atleast $2^|n|$ hypothesis

What are the 3 approaches to multi-label classification?

- binary relevance, create a decision tree for each label
- power labeling, create a deceision tree for every element in the powerset of the labe,s
- vector encoding, encode the labels as vector and learn a tree to predict the vectors.

What is a loss function?

- A loss function describes the quality of the predictions, the higher the loss function, the worse the predictions. The problem with this is that we usually train on sample data and if we try to minimize the loss function during training we may get overfitting. The solution for this is regularization