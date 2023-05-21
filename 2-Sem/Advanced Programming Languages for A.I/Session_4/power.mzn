include "globals.mzn"; 

array[1..4] of var 1..1000:x;


constraint 
        pow(x[1],3) + pow(x[2],3) = pow(x[3],3) + pow(x[4],3) /\
        forall(i in 1..4, j in 1..4 where i != j)(
          x[i] != x[j]
        )

