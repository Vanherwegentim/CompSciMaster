
int: bags;

% var 0..16:bag1;
% var 0..17:bag2;
% var 0..23:bag3;
% var 0..24:bag4;
% var 0..39:bag5;
% var 0..40:bag6;
int: bag1=16;
int: bag2=17;
int: bag3=23;
int: bag4=24;
int: bag5=39;
int: bag6=40;


array[1..6] of var 0..bags:num_bags;



solve satisfy;

constraint 
    bag1 * num_bags[1] + bag2 * num_bags[2] + bag3 * num_bags[3] + bag4 * num_bags[4]
    + bag5 * num_bags[5] + bag6 * num_bags[6] = 100;
