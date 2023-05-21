include "globals.mzn"; 

int:n;
set of int:size = 1..n;

array[size,size] of var 1..n*n:sq;

function int:sums(array[size] of int:x) = sum([y|y in x]);

int: num = ( n * (n*n + 1)) div 2;
  

constraint
   all_different(sq) /\
   forall (i in 1..n)(
        sum(j in 1..n)(sq[i,j]) = num
          )
   /\
   forall (i in 1..n)(
        sum(j in 1..n)(sq[j,n]) = num
          )
   /\
        sum(i in 1..n)(sq[i,i]) = num
        /\
        sum(i in 1..n)(sq[i,n+1-i]) = num;