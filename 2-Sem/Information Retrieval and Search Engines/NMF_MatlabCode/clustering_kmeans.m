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
  %% K-Means (Matlab default - squared Euclidean distance measure and the k-means++ algorithm for cluster center initialization)
  ind = kmeans(V,k)

  %% Assign actual clusters
  for i = 1:max(ind)
      disp(['Cluster ', int2str(i)])
      cluster{i} = find(ind == i);
      disp(cluster{i})
  end

