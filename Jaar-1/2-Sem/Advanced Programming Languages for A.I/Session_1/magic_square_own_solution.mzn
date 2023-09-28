include "globals.mzn"; 

int: n;
set of int: size = 1..n;

array[size, size] of var 1..n^2: square;

constraint forall(i in size)(
  sum(j in size)(square[i,j]) = n*((n^2)+1) div 2
);

constraint forall(i in size)(
    sum(j in size)(square[j,i]) = n*((n^2)+1) div 2
);

constraint forall(i in size)(
    sum(j in size)(square[j,j]) = n*((n^2)+1) div 2 %Diagonal but I don't know how to
);

constraint
  all_different(square);
  
solve satisfy;