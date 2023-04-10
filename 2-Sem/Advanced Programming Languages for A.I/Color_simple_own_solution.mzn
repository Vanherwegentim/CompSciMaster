 
include "globals.mzn"; 
int: n = 5;

set of int: countries = 1..6;
array[countries] of var 1..n: x = [belgium, denmark, france, germany, netherlands, luxembourg];

var 1..n: belgium;
var 1..n: denmark;
var 1..n: france;
var 1..n: germany;
var 1..n: netherlands;
var 1..n: luxembourg;

solve minimize(n);
constraint
    belgium != france /\
    france != luxembourg /\
    france != germany /\
    luxembourg != germany /\
    luxembourg != belgium /\
    belgium != netherlands /\
    belgium != germany /\
    germany != netherlands /\
    germany != denmark
;