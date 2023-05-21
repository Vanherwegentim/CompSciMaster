winclude "globals.mzn"; 

array[1..10] of var 1..100: num_list;

set of int: nums = 1..100;

var set of nums:s;
var set of nums:t;
solve satisfy;

set of int: a = {2,5,87,9,11,14,17,19,24,28};

function var int: sum_of_set(var set of int:x) = sum([y | y in x]);

constraint
  s subset a
  /\
  t subset a
  /\
  sum_of_set(s)= sum_of_set(t)
  /\
  forall (i in s) (
    not (i in t)
  )

;
