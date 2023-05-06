include "globals.mzn"; 

set of int:size;
array[1..6] of var size: x = [g1,g2,g3,g4,g5,g6];

var size: g1;
var size: g2;
var size: g3;
var size: g4;
var size: g5;
var size: g6;

solve satisfy;
constraint 
  g6=1 /\
  g6 = (g5 div 2) - 1 /\
  g5 = (g4 div 2) - 1 /\
  g4 = (g3 div 2) - 1 /\
  g3 = (g2 div 2) - 1 /\
  g2 = (g1 div 2) - 1;
  
  
