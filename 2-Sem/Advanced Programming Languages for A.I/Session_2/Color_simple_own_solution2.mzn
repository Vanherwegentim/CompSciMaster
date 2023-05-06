
int:n= 6;
array[1..6] of var 1..n: x = [belgium,denmark,france,germany,netherlands,luxembourg];
int: num_nodes; 
array[1..num_nodes,1..2] of int: graph;

var 1..n: belgium;
var 1..n: denmark;
var 1..n: france;
var 1..n: germany;
var 1..n: netherlands;
var 1..n: luxembourg;

constraint
  forall(i in 1..9) (
    x[graph[i,1]] != x[graph[i,2]]
    );

solve minimize(n);