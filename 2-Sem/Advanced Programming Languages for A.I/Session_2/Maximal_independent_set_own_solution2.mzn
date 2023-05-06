include "globals.mzn"; 

int:n;
array[1..6] of set of int:v;
array[1..6] of var 0..1: x;

solve maximize sum(x);

constraint
  forall(i,j in 1..6 where j>i) (
       x[i] = 1 /\ x[j] =1 -> (v[i] intersect v[j] = {})
      
      
      )
      ;