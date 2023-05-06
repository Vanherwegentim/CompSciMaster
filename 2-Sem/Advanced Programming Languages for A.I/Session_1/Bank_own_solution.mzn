include "globals.mzn"; 

set of int: nums = 1..4;
array[nums] of var 1..9: x = [a,b,c,d];

var 1..9:a;
var 1..9:b;
var 1..9:c;
var 1..9:d;


solve satisfy;

constraint
  all_different(x) /\
  (c *10 + d) = 3*(a*10 +b) /\
    (d *10 + a) = 2*(b*10 +c) ;
