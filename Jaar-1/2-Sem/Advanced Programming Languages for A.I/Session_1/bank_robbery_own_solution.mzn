
set of int:bin = 0..1;
array[1..6] of var bin: suspects = [artie,bill,crackitt,dodgy, edgy,fingers];

var 0..1:artie;
var 0..1:bill;
var 0..1:crackitt;
var 0..1:dodgy;
var 0..1:edgy;
var 0..1:fingers;

constraint
         
  (artie = 0 \/ artie = 1) /\
  (crackitt = 1 \/ bill = 1) /\
  (crackitt = 0 \/ crackitt = 1) /\
  ((crackitt = 1 -> bill = 1) \/ dodgy = 1) /\
  edgy = 0 /\
  ((artie = 1 <-> dodgy =1) \/ fingers = 1) /\
  sum(suspects) = 2
;