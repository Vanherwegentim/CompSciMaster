include "globals.mzn";
int:num_people;
array[1..num_people] of set of 1..8:compatible;
array[1..num_people] of var 0..8: receiver;

var 0..num_people: exchanges = sum([bool2int(receiver[i] > 0) | i in 1..num_people]);

solve :: int_search(receiver, first_fail, indomain_min, complete) maximize exchanges;


constraint
       all_different_except_0(receiver) /\ 
       forall(i in 1..num_people) (
          receiver[i] != 0 -> receiver[i] in compatible[i] /\
          receiver[receiver[i]] > 0
       )

