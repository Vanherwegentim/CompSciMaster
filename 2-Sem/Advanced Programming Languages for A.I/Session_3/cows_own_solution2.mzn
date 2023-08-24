include "globals.mzn";
array[1..5] of var set of 1..25:sons;

solve satisfy;

function var int:sumset(var set of int:x)= sum([i|i in x]);
constraint
      
      forall(i in 1..5)(
        sumset(sons[i]) = 65
        
      ) /\
       forall(i,j in 1..5 where i!=j)(
        disjoint(sons[i],sons[j])
       ) /\
       card(sons[1]) = 7 /\
              card(sons[2]) = 6 /\
                     card(sons[3]) = 5 /\
                            card(sons[4]) = 4 /\
                                   card(sons[5]) = 3;