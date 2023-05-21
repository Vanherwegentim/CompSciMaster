include "globals.mzn"; 

set of int:A = {5,6,15,21,86,47,68,23,52,44};
var set of 1..100:S;
var set of 1..100:T;

function var int: sumset(var set of int:P) = sum([y | y in P]);
constraint
      S subset A /\
      T subset A /\
      sumset(S) = sumset(T) /\
      disjoint(S,T); 