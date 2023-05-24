clear all; clc; close all;

threes = load('threes.mat','-ascii');
threes_M = mean(threes);
%imagesc(reshape(threes(1,:),16,16),[0,1])
imagesc(reshape(threes_M,16,16),[0,1]);
%% 


c = 50;
X_reconstructed = zeros(c);
for i = 1:c
    [V,D] = eigs(cov(threes),i);

    X_reduced = threes * V;
    X_reconstructed = X_reduced * V' ;
    rmse(i) = sqrt( mean( mean( mean((threes - X_reconstructed).^2))));

end
imagesc(reshape(X_reconstructed(1,:),16,16),[0,1])

plot(diag(D));
%%
sum = cumsum(D);
flipped = flip(sum(end, :));
sumsum = cumsum(flipped);
%%

figure;
plot(sumsum, 'b');
title('Cumsum');
figure;
plot(diag(D), 'g');
title('Eigenvalues');
figure;
plot(rmse, 'r');
title('reconstruction error');


