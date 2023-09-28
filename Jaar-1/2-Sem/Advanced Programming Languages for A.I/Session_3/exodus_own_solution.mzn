include "globals.mzn"; 
int: n = 5;

set of int: countries = 1..5;
array[countries] of var 1..n: x = [ kazakhstan, lithuania, morocco, yemen, ethiopia];

var 1..n: kazakhstan;
var 1..n: lithuania;
var 1..n: morocco;
var 1..n: yemen;
var 1..n: ethiopia;

set of int: names = 1..5;
array[names] of var 1..n: y = [bernice, carl, debby, sammy, ted];

var 1..n: Bernice;
var 1..n: Carl;
var 1..n: Debby;
var 1..n: Sammy;
var 1..n: Ted;

set of int: nums = 1..5;
array[nums] of var 1..n: z = [three, five, seven, eight, ten];

var 1..n: three;
var 1..n: five;
var 1..n: seven;
var 1..n: eight;
var 1..n: ten;

solve satisfy;

constraint
    debby = lithuania /\
    