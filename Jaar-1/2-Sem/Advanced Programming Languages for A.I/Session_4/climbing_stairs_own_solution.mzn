
int:n;
int:m;

array[1..n] of var 0..m:stairs;

solve satisfy;

constraint
    sum(stairs) = n /\
    forall(i in 2..n) (
    if stairs[i-1] = 0 then stairs[i] = 0  else true endif
   )
   ;