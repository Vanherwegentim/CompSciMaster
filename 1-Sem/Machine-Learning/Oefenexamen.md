# Oefenexamen

### Question 1

Predictive because we want to be able to predict the name given a face and not look for relationships within the data.

Supervised because we train the software on labeled data, we give it a picture of a face and the name corresponding to that face.

Classification because we want to classify a face as a name.





Decision trees would not be suitable because decision trees are build on splits with numerical values and would not be able to process the raw data in pictures

Rule learning is also not suitable. There is no rule in the form if then else that we can learn that can process the data of a picture

Association rules look for the underlying relationship or patterns in data and doesn't return a model that can predict something so it also wouldn't be suitable

Neural networks would be very suitable because we have seen that they handle raw data very well and will give us a model that can accurately predict it

Support vector machines would be suitable if they could find process the raw data to find a hyperplane that correctly separate all the data points

Nearest neighbor methods could be used if the raw input of pictures could be converted to data that can be used in those methods. We are predicting names based on similarity so in theory it should be possible.

*Q-learning might be possible if we can create good reward function to create the behaviour we want to see*

Inductive logic programming is not suitable because how would we convert the raw data of pictures to logic clauses





### Question 2

3

1

2

4



### Question 3

Given the output of A we could make a classifier that reverse every predictions classifier A makes. If we have a prediction of A that says it is negative, B says it is positive and vice versa.

Based on the given example the accuracy of B will be 1. accuracy= |positive predictions|/|actual positives|



### Question 4

Naive bayes assumes that some class is independent give some other class (conditionally independent). This assumption is not realistic. This does affect the quality of predictions but not enough to use a classifier that will require more computational power because most uses of text classification do not require a rediculous amount of accuracy.



### Question 5

a) High, given that we need to define each nominal value of x_1...x_n for the 3 possible values of 

b) Medium, We can cut down a lot because we know which attributes are independent on which

c) Low, In naive bayes we say that everything is independent given some class so we will have a lot less information to be saved

###  Question 6

The problem with reinforcement learning and especially Q-learning is that there is no generalization. It learns some q-values in an environment for some state-action pairs but given a new environment this is useless. We can solve this by taking these state-action pairs for which we have the Q-values and put this in a learner. In this case regression trees. Regression trees are very fast and can thus make reinforcement learning more efficient.



### Question 7

![img](https://cdn.discordapp.com/attachments/1041382555326369834/1062402218449449012/20230110_151101.jpg)



### Question 8

a) argmax P(weather=R|Beer=D)P(place=I|Beer=D)P(Beer=D) = 3/32

​	argmax P(weather=R|Beer=J)P(place=I|Beer=J)P(Beer=J) = 2/32

​	argmax P(weather=R|Beer=H)P(place=I|Beer=H)P(Beer=H) = 2/32

Duvel voor naive bayes.

b) since 2,5,7 are closest but they all predict something different we can draw no conclusions.'





### Question 9

poop



### Question 10

Upper bound: 4

lower bound: 0



We only have 2 nodes with each one variable to classify the boolean variables. In a graph this would mean we can only draw max 2 lines to classify every variable. This means that we can only shatter 3 variables at once. Because we can always draw a line over the 2 variables with same boolean and thus splitting the booleans correctly



