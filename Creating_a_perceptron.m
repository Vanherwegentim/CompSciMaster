%% Create a perceptron and train it with examples
%Creating new data and plotting this

P = [-1.5 -1.5 +1.3 -1.1 -2; ...
     -1 +2 -2 +1.0 2];
T=[0 1 0 1 0];
plotpv(P,T);

%% Create a network using the previous information
% PERCEPTRON creates a new network which is then configured with the input
% and target data which results in initial values for its weights and bias.
% (Configuration is normally not necessary, as it is done automatically
% by ADAPT and TRAIN.)

net = perceptron;
net = configure(net,P,T);

%% Add the neuron's initial attempt at classification to the plot
% The initial weights are set to zero, so any input gives the same output
% and the classification line does not even appear on the plot.
hold on
linehandle = plotpc(net.IW{1},net.b{1});

%% Adapt returns a new network object that performs as a better classifier, the 
% network output, and the error.  This loop adapts the network and plots
% the classification line, until the error is zero.

E = 1;
while (sse(E))
   [net,Y,E] = adapt(net,P,T);
   linehandle = plotpc(net.IW{1},net.b{1},linehandle);
   drawnow;
end

%% Add the points back to the graph

hold on;
plotpv(P,T);
plotpc(net.IW{1},net.b{1});
hold off;
