include "globals.mzn";

% Use this editor as a MiniZinc scratch book
% array[1..5] of var 1..25:son1;
% array[1..5] of var 1..25:son2;
% array[1..5] of var 1..25:son3;
% array[1..5] of var 1..25:son4;
% array[1..5] of var 1..25:son5;

array[1..5] of var set of 1..25:sons = [son1,son2,son3,son4,son5];
var set of 1..25:son1;
var set of 1..25:son2;
var set of 1..25:son3;
var set of 1..25:son4;
var set of 1..25:son5;

function var int:sumset(var set of int:x) = sum([y|y in x]);

int: n = 65;
solve satisfy;

constraint
      forall(i in 1..5)(
        sumset(sons[i]) = n
      )
      /\
      forall(i in 1..5, j in 1..5 where i !=j)(
          disjoint(sons[i],sons[j])
      )