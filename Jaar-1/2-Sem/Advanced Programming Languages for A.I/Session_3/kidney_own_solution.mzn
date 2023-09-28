% Use this editor as a MiniZinc scratch book
include "globals.mzn";

int: num_people;
set of int:people = 1..num_people;
array[people] of set of int: compatible;

array[people] of var 0..num_people: gets_kidney;

solve maximize(sum(gets_kidney));

constraint

   alldifferent_except_0(gets_kidney)
   /\
   forall(i in people) (
     (gets_kidney[i] = 0)
     \/
     (
        gets_kidney[i] in compatible[i] /\
        gets_kidney[gets_kidney[i]] > 0 % the giver must also receive a kidney
     )
   )
;