clear
clc
close all

%%%%%%%%%%%
%algorlm.m
% A script comparing performance of 'trainlm' and 'trainbfg'
% trainbfg - BFGS (quasi Newton)
% trainlm - Levenberg - Marquardt
%%%%%%%%%%%

% Configuration:
alg1 = 'trainlm';% First training algorithm to use
alg2 = 'trainbfg';% Second training algorithm to use
alg3 = 'trainbr';% Second training algorithm to use

H = 500;% Number of neurons in the hidden layer
delta_epochs = [1,14,985];% Number of epochs to train in each step
epochs = cumsum(delta_epochs);

%generation of examples and targets
dx=0.05;% Decrease this value to increase the number of data points
x=0:dx:3*pi;y=sin(x.^2);
sigma=0.2;% Standard deviation of added noise
yn=y+sigma*randn(size(y));% Add gaussian noise
t=yn;% Targets. Change to yn to train on noisy data

%creation of networks
net1=feedforwardnet(H,alg1);% Define the feedfoward net (hidden layers)
net2=feedforwardnet(H,alg2);
net3=feedforwardnet(H,alg3);

net1=configure(net1,x,t);% Set the input and output sizes of the net
net2=configure(net2,x,t);
net3=configure(net3,x,t);

net1.divideFcn = 'dividetrain';% Use training set only (no validation and test split)
net2.divideFcn = 'dividetrain';
net3.divideFcn = 'dividetrain';

net1=init(net1);% Initialize the weights (randomly)
net2.iw{1,1}=net1.iw{1,1};% Set the same weights and biases for the networks 
net2.lw{2,1}=net1.lw{2,1};
net2.b{1}=net1.b{1};
net2.b{2}=net1.b{2};
net3.iw{1,1}=net1.iw{1,1};% Set the same weights and biases for the networks 
net3.lw{2,1}=net1.lw{2,1};
net3.b{1}=net1.b{1};
net3.b{2}=net1.b{2};

%training and simulation
net1.trainParam.epochs=delta_epochs(1);  % set the number of epochs for the training 
net2.trainParam.epochs=delta_epochs(1);
net3.trainParam.epochs=delta_epochs(1);

net1=train(net1,x,t);   % train the networks
net2=train(net2,x,t);
net3=train(net2,x,t);

a11=sim(net1,x); a21=sim(net2,x); a31=sim(net3,x);
% simulate the networks with the input vector x

net1.trainParam.epochs=delta_epochs(2);
net2.trainParam.epochs=delta_epochs(2);
net3.trainParam.epochs=delta_epochs(2);

net1=train(net1,x,t);
net2=train(net2,x,t);
net2=train(net2,x,t);

a12=sim(net1,x); a22=sim(net2,x);a32=sim(net3,x);

net1.trainParam.epochs=delta_epochs(3);
net2.trainParam.epochs=delta_epochs(3);
net3.trainParam.epochs=delta_epochs(3);

net1=train(net1,x,t);
net2=train(net2,x,t);
net3=train(net2,x,t);

a13=sim(net1,x); a23=sim(net2,x); a33=sim(net3,x);
%% 

%plots
figure
subplot(3,1,1);
plot(x,t,'bx',x,a11,'r',x,a21,'g', x,a31,'y'); % plot the sine function and the output of the networks
title([num2str(epochs(1)),' epochs']);
legend('target',alg1,alg2,alg3,'Location','north');
subplot(3,1,2);
plot(x,t,'bx',x,a12,'r',x,a22,'g',x,a32,'y');
title([num2str(epochs(2)),' epoch']);
legend('target',alg1,alg2,alg3,'Location','north');
subplot(3,1,3);
plot(x,t,'bx',x,a13,'r',x,a23,'g',x,a33,'y');
title([num2str(epochs(3)),' epoch']);
legend('target',alg1,alg2,alg3,'Location','north');
%% 

figure
subplot(3,3,1);
postregm(a11,y); % perform a linear regression analysis and plot the result
subplot(3,3,2);
postregm(a21,y);
subplot(3,3,3);
postregm(a31,y);
subplot(3,3,4);
postregm(a12,y);
subplot(3,3,5);
postregm(a22,y);
subplot(3,3,6);
postregm(a32,y);
subplot(3,3,7);
postregm(a13,y);
subplot(3,3,8);
postregm(a23,y);
subplot(3,3,9);
postregm(a33,y);
