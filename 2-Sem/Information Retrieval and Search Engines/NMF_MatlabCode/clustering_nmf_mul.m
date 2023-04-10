%% Document-term matrix (rows are documents)
V = [ 1 1 1 0 0 0 0 0
      1 0 1 0 0 0 0 0
      0 0 0 1 1 1 0 0
      1 0 0 0 0 0 1 1
      1 0 0 0 0 0 0 1
      1 1 0 0 0 0 0 0
      1 0 0 0 0 0 1 0
      0 0 0 1 0 1 0 0
      0 0 0 0 1 1 0 0 ];
  
%% Non-negative matrix factorization
  % Number of clusters
  k = 3
  options = statset('MaxIter',500) 
  % NMF using the Euclidean distance
  [Ws,H] = nnmf(V,k,'algorithm','mult','options',options)
  
%% Clustering
  % Pick the max along each row, together with the index where the max occurs
  [cen,ind] = max(Ws,[],2)
  cluster = cell(1,k); 
  for i = 1:max(ind)
      disp(['Cluster ', int2str(i)])
      cluster{i} = find(ind == i);
      disp(cluster{i})
  end
