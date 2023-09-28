
% Original matrix V
V = [ 1 1 1 0 0 0 0 0
      1 0 1 0 0 0 0 0
      0 0 0 1 1 1 0 0
      1 0 0 0 0 0 1 1
      1 0 0 0 0 0 0 1
      1 1 0 0 0 0 0 0
      1 0 0 0 0 0 1 0
      0 0 0 1 0 1 0 0
      0 0 0 0 1 1 0 0 ];


options = statset('MaxIter',500,'Display','iter');

%% Non-negative matrix factorization
  % Number of dimensions
  k = 3;
  % NMF using the Euclidean distance
  [Ws,H] = nnmf(V,k,'algorithm','mult','options',options)

