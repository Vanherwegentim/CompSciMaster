array[1..4] of var 1..711:item;

constraint
        item[1] + item[2] + item[3]+item[4] = 711 /\
        item[1]* item[2]* item[3]*item[4] = 711 *100*100*100;
        
        
        
constraint symmetry_breaking_constraint(
            item[1]<item[2] /\ item[2] < item[3] /\ item[3] < item[4]
            );
solve::int_search(item,first_fail,indomain_min,complete) satisfy;
        