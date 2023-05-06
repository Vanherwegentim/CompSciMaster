include "globals.mzn"; 
%assignment en oplossing zijn bagger, de oefening is niet opgelost denk ik
int:num_people; %number of people on waiting list
set of int:people = 1..num_people;

array[people] of set of int: compatible;%set of people each person can donate to

array[people] of var 0..num_people: gets_kidney; %people that get kidneys from who

    %Only one kidney can be given away
constraint
    alldifferent_except_0(gets_kidney);
    
    %Anyone who donates a kidney must also receive one
% constraint  
%    forall(i in people)(
%       if gets_kidney[i] != 0 then gets_kidney[gets_kidney[i]] != 0 endif
%      )
%      ;
     %People can only receive from certain people
constraint forall(i in people) (
        forall(j in compatible[i])(
            gets_kidney[j] = i \/ gets_kidney[j] = 0
      )
  );

    
solve maximize sum(gets_kidney);
