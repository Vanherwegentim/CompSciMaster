%%%%%%%%%%%
% rep2.m
% A script which generates n random initial points 
%and visualises results of simulation of a 2d Hopfield network 'net'
%%%%%%%%%%

clear
clc
close all
a = {[0;0]}
T = [1 1; -1 -1; 1 -1]';
net = newhop(T);
net.layers{1}
view(net);
n=30;
timestep = 50;
%for i=1:n
for i=1:1
    %a={rands(2,1)}                  % generate an initial point
    a(i)
    [y,Pf,Af] = sim(net,{1 timestep},{},a(i));   % simulation of the network for 50 timesteps              
    record=[cell2mat(a(i)) cell2mat(y)];   % formatting results  
    start=cell2mat(a(i));                  % formatting results 
    plot(start(1,1),start(2,1),'bx',record(1,:),record(2,:),'r'); % plot evolution
    hold on;
    plot(record(1,timestep),record(2,timestep),'gO');  % plot the final point with a green circle
%end
end
legend('initial state','time evolution','attractor','Location', 'northeast');
title('Time evolution in the phase space of 2d Hopfield model');
