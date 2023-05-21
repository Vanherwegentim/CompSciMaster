clear all; clc; close all;

threes = load('threes.mat','-ascii');
%imagesc(reshape(threes(1,:),16,16),[0,1])
%imagesc(reshape(threes_M,16,16),[0,1])

%plot(diag(D));

c = 256;

for i = 1:c
    [V,D] = eigs(cov(threes),i);

    X_reduced = threes * V;
    X_reconstructed = X_reduced * V' ;
    imagesc(reshape(X_reconstructed(1,:),16,16),[0,1]);
    rmse(i) = sqrt( mean( mean((threes - X_reconstructed).^2)));

end


sum = cumsum(D)
sumAgain = cumsum(sum(end, :));


figure;
tiledlayout(1,3);

nexttile;
plot(diag(D), 'LineWidth', 2);
title('Eigenvalues');
axis([1 c 0 inf]);

nexttile;
plot(rmse, 'LineWidth', 2);
title('reconstruction error');
axis([1 c 0 inf]);

nexttile;
plot(sumAgain, 'LineWidth', 2);
title('cumsum of eigenvalues');
axis([1 c 0 inf]);
