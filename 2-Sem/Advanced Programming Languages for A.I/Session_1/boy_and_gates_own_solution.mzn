include "globals.mzn"; 

set of int:guards = 1..6;
array[guards] of var int: x;

constraint
     all_different(x) /\
     x[6] = 1 /\
     forall (i in 1..5) (
        x[i] = (x[i+1]+1) *2 
     );