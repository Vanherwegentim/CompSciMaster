**What is a decision variable?**

Decision variables are implicitly existentially quantified: the aim is to find satisfying optimal values in their domain.

**What is consistency?**

Consistency refers to the property of a constraint being satisfied by the current variable assignments. It ensures that the constraints are not violated and that the model is in a valid state

**What is a viewpoint?**

A viewpoint is a choice of decision variables

- Has an impact on the way constraints are formulated
- Try investigate alternatives and try to formulate the constraints
- Can have an impact on performance and readability



**What is a permutation problem?**

In a permutation problem, we have as many values as decision variables and each variable takes an unique value. This <X,D>  with X the set of decision variables and D the domain of decision variables such that X and D have the same number of elements, say n. If we then switch the role of values and variables:  <D,X> is also a viewpoint

**What are implied constraints?**

An implied constraint, also called a redundant constraint, is a constraint that logically follows from other constraints. 

Benefit: Solving may be faster, without losing any solutions

**What are redundant decision variables?**

A redundant decision variable is a decision variable that represents information that is already represented by some other decision variables. It reflects a different viewpoint. Benefit: Easier modelling of some constraints or faster solving, or both

**What is channeling constraint and give an example?**

A channelling constraint establishes the coherence of the values of mutually redundant decision variables.

**What is pre-computation and give an example**

Some problems are solved more efficiently if we pre-compute all possible objective values.

**Explain symmetry and give an example**

Symmetry refers to the existence of redundant or equivalent solutions in a constraint satisfaction problem. Symmetry occurs when multiple solutions to a problem are essentially the same, except for some permutation or rearrangement of values or variables

Full symmetry: any permutation preserves solutions.

Partial symmetry: any piecewise permutation preserves solutions

Index symmetry: any permutation of slices of an array of decision variables preserves solutions



**What is symmetry breaking and what kinds are there?**

While solving, keep ideally one member per symmetry class, as this may make a problem "less intractable" We can do this by: 

Symmetry breaking by reformulation: The elimination of symmetries detectable in model 

Static symmetry breaking (SSB): the elimination of symmetric solutions by constraints 

Dynamic symmetry breaking: the elimination of symmetric solutions by search, this is beyond the scope of this topic

**What is a propagator and what kinds are there?**

A propagator for a predicate $\gamma$, removes from the current domains of the variables of a $\gamma$ -constraint the values that cannot be part of a solution to that constraint. 

Domain-consistency propagator deletes all impossible values from the domains 

Bounds-consistency propagator only deletes all impossible min and max values from the domains

**What is variable selection strategy and what kinds are there?**

The variable selection strategy has an impact on the size of the search tree, especially of the constraints are processed with propagation at every node of the search tree or if the whole search tree is explored. 

First-fail principle To succeed: first try where you are most likely to fail. 

Best-first principle First try a domain part that is most likely, if not guaranteed, to have values that lead to solutions. This may be like how one would make the greedy choice in a greedy algorithm for the problem at hand.





