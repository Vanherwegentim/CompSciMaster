
include "globals.mzn"; 
int: num_movies;

array[1..num_movies,1..2] of int: data;
array[1..num_movies] of string: movies;

array[1..num_movies] of var 0..1: watch_movie;
var 0..num_movies: amount_of_movies = sum(watch_movie);

predicate no_overlap(int:ts1, int: te1, int:ts2, int:te2)=
    ts2 > te1 
    \/
    ts1 > te2
; 

solve maximize amount_of_movies;

constraint
  forall (i, j in 1..num_movies where i< j) (
   (watch_movie[i] = 1 /\ watch_movie[j]=1) -> no_overlap(data[i,1],data[i,2],data[j,1],data[j,2]));

