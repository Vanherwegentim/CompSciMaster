include "globals.mzn"; 

int:n;
array[1..6] of int: x = [16, 17, 23, 24, 39, 40];
array[1..6] of var 0..n: amount = [bag1,bag2,bag3,bag4,bag5,bag6];

var 0..n: bag1;
var 0..n: bag2;
var 0..n: bag3;
var 0..n: bag4;
var 0..n: bag5;
var 0..n: bag6;
solve satisfy;

constraint
    x[1]*bag1 + x[2] *bag2 + x[3]*bag3 + x[4] *bag4 + x[5] * bag5 + x[6] *bag6 = 100;
  

