
% Original matrix V
V = [2 4 0
     5 2 0
     0 1 6 
    ];

W0 = [0.33 0.33
      0.33 0.33
      0.33 0.33
     ];

H0 = [0.5 0.5 0.5
      0.5 0.5 0.5
     ];


options = statset(5,2,'Display','iter');

%% Non-negative matrix factorization
  % Number of dimensions
  k = 2;
  % NMF using the Euclidean distance
  [Ws,H] = nnmf(V,k,'algorithm','mult','w0',W0,'h0',H0,'options',options)




 % Load the text data
documents = {'This is the first document.', 'This is the second document.', 'This is the third document.'};

% Preprocess the text
documents = lower(documents);
documents = erasePunctuation(documents);
documents = stopWords(documents);

% Tokenize the text
tokens = regexp(documents, '\w+', 'match');

% Create a vocabulary
vocabulary = unique([tokens{:}]);

% Create the matrix
matrix = sparse(numel(documents), numel(vocabulary));
for i = 1:numel(documents)
    docTokens = tokens{i};
    for j = 1:numel(docTokens)
        token = docTokens{j};
        k = find(strcmp(vocabulary, token));
        matrix(i,k) = matrix(i,k) + 1;
    end
end

