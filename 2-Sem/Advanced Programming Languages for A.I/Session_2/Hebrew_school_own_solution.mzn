include "globals.mzn"; 

set of int: num = 1..5;
array[1..5] of var num: names = [bernice, carl, debby, sammy, ted];
var num: bernice;
var num: carl;
var num: debby;
var num: sammy;
var num: ted;
array[1..5] of var num: age = [three,five,seven,eight,ten];
var num:three;
var num:five;
var num:seven;
var num:eight;
var num:ten;
array[1..5] of var num: countries = [ethiopia, kazakhstan, lithuania, morocco, yemen];
var num: ethiopia;
var num: kazakhstan;
var num: lithuania;
var num: morocco;
var num: yemen;
array[1..5] of var num: story = [bush, captivity, moses,passover, tencom];
var num: bush;
var num: captivity;
var num: moses;
var num: passover;
var num: tencom;

solve satisfy;

constraint
  all_different(age) /\
    all_different(countries) /\
    all_different(story) /\
    all_different(names) /\

  debby = lithuania /\
  passover+1 = bernice /\
  yemen < ethiopia /\
  morocco + 2 = ted /\
  sammy + 2 = moses /\
  carl = captivity /\
  bush + 2 = kazakhstan;
  
