include "globals.mzn"; 

var set of 1..100:t;
var set of 1..100:s;

solve satisfy;

set of int: a = {2,5,87,9,11,14,17,19,24,28};
function var int: sum_set(var set of int:x) = sum([y|y in x]) ;

constraint
    t subset a /\
    s subset a /\
    sum_set(t) = sum_set(s) /\
    disjoint(t,s)
  ;
    