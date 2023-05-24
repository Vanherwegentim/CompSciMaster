%%%%%%%%%%%
% rep3.m
% A script which generates n random initial points for
% and visualise results of simulation of a 3d Hopfield network net
%%%%%%%%%%
T = [1 1 1; -1 -1 1; 1 -1 -1]';
net = newhop(T);
a = {[0; 0; 0]};
a(1)
n=10;
for i=1:1
    %a={rands(3,1)};                        % generate an initial point                   
    [y,Pf,Af] = sim(net,{1 100},{},a(1));       % simulation of the network  for 50 timesteps
    record=[cell2mat(a(1)) cell2mat(y)];       % formatting results
    start=cell2mat(a(1));                      % formatting results 
    plot3(start(1,1),start(2,1),start(3,1),'bx',record(1,:),record(2,:),record(3,:),'r');  % plot evolution
    hold on;
    plot3(record(1,100),record(2,100),record(3,100),'gO');  % plot the final point with a green circle
end
grid on;
legend('initial state','time evolution','attractor','Location', 'northeast');
title('Time evolution in the phase space of 3d Hopfield model');
