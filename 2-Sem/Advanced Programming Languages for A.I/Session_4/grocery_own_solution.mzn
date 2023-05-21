var 1..711: item1;
var 1..711: item2;
var 1..711: item3;
var 1..711: item4;
int: totalPayment = 711;
int: multiplication = 711*100*100*100;

constraint item1 >= 0 /\ item1 <= 711;  % Item prices should be non-negative
constraint item2 >= 0 /\ item2 <= 711;
constraint item3 >= 0 /\ item3 <= 711;
constraint item4 >= 0 /\ item4 <= 711;

constraint totalPayment = item1 + item2 + item3 + item4;  % Total payment is the sum of item prices


constraint item1 * item2 * item3 * item4 = multiplication;

constraint item1< item2/\ item2<item3 /\ item3<item4;
solve satisfy;

output ["Item 1: ", show(item1), "\n"];
output ["Item 2: ", show(item2), "\n"];
output ["Item 3: ", show(item3), "\n"];
output ["Item 4: ", show(item4), "\n"];
output ["Total Payment: ", show(totalPayment), "\n"];
