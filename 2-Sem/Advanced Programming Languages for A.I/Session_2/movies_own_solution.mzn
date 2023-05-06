
int: num_movies; 
array[1..9] of var 0..1: movies_s;
array[1..9,1..2] of int: data;

solve maximize(sum(movies_s));

constraint 
    forall(i, j in 1..num_movies where i < j) (

     (movies_s[i] = 1 /\ movies_s[j] = 1) -> (data[i,2] < data[j,1] \/ data[i,1] > data[j,2])
    )
   
;
