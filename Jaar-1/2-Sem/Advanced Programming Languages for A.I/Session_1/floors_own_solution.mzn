include "globals.mzn"; 

set of int: floors = 1..5;
array[floors] of var 1..5: x = [baker, cooper, fletcher, miller, smith];

var 1..5: baker;
var 1..5: cooper;
var 1..5: fletcher;
var 1..5: miller;
var 1..5: smith;

solve satisfy;

constraint
  baker != 5 /\
  cooper != 1 /\
    fletcher != 1 /\ fletcher != 5 /\
    miller > cooper /\
    abs(smith-fletcher) > 1 /\
    abs(fletcher - cooper) > 1 /\
    all_different(x);