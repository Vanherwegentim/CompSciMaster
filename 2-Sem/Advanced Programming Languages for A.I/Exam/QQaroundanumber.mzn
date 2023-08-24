% Given a Q*Q matrix M, the value domain is 1..Q*Q, design a minizinc model to find a %solution such that, every M[i,j] value should have its adjacent value at its neighbor. %e.g., M[3,3]=15, then number 14 or 16 should be at one of the following position: %M[2,2],M[2,3],M[2,4],M[3,2],M[3,4],M[4,2],M[4,3],M[4,4].
include "globals.mzn";

int:Q=5;

set of int:dom = 1..Q;
array[dom,dom] of var 1..Q*Q: m;

constraint
    forall(i,j in dom)(
        exists(x in i-1..i+1 where x >0 /\ x <Q+1, y in j-1..j+1 where y > 0 /\ y < Q+1)(
            abs(m[x, y] - m[i, j]) = 1
        
      
      )
    
    )
    ;
    
constraint 
      all_different(m);
    

