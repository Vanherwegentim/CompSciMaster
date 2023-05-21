# Advanced Programming Languages for AI



### **Modelling Concepts**

- **Decision variable (also just variable)** is an existentially quantified unknown of a problem
- **Domain** of a variable x, usually denoted by dom(x), is the set of values that x can take.
- **Variable expression** takes a value that depends on the value of one or more decision variables.
- **Parameter** has a value from a problem description
- **Constraint** is a restriction on the values that its variables can take conjointly; equivalently, it is a Boolean-valued variable expression that must be true
- **Objective function** is a numeric variable expression whose value is to be minimized or maximized.



**Types available in Minizinc:**

- ```int:``` integer
- ```bool:``` boolean
- ```enum:``` enumeration
- ```float:``` floating-point number
- ```string:``` string
- ```set of T:``` set of elements of type T 
- ```array[P] of T:``` possibly multidimensional array of elements of type T, which is not an array.



**Constraint predicates** are predicates that already exist such as ```all_different(), count,etc... ``` . They also go by the name, **global-constraint predicates**.



- **all_different(A)** holds iff all elements of A take different values
- **nvalue(m,A)** holds iff all elements of A take m different values
- **global_cardinality(vars,values,card)** hold iff vars contains all the elements of values with the corresponding amounts in card.
- **circuit(s)**: holds iff the arcs $v \rightarrow S[v]$ form a **Hamiltonian circuit**: each vertex is visited exactly once.
- **subcircuit(s)** holds iff circuit($S$') for exactly one possibly empty but non-singleton subarray $S'$ of $S$, and $S[v]=v$ for all the other vertices. 
- **lex_lesseq(A,B)** holds iff A is lexicographically at most equal to B





#### **Viewpoints**

A viewpoint is a choice of decision variables

- This choice has a big impact on the way the constraints are/can be formulated
- Good practice: investigate alternatives and try to formulate the constrains



**Permutation problem**

In a permutation problem, we have as many values as decision variables and each varaiable takes an unique value.



#### Implied constraints

An implied constraint, also called a redundant constraint, is a constraint that logically follows from other constraints.

**Benefit:**

- Solving may be faster



#### Redundant decision variables

A redundant decision variable is a decision variable that represents information that is already represented by some other decision variables. It reflects a different viewpoint.

**Benefit:**

- Easier modelling of some constraints or faster solving, or both



#### Channelling constraint

A channelling constraint establishes the coherence of the values of mutually redundant decision variables.



#### Global inverse constraint

An **inverse(f,invf)** constraint on two arrays of int variables holds off f and invf represent inverse functions.



#### Pre-computation

Some problems are solved more efficiently if we pre-compute all possible objective values.

![image-20230520190841103](img/image-20230520190841103.png)

#### Symmetry

Symmetry refers to the existence of redundant or equivalent solutions in a constraint satisfaction problem. Symmetry occurs when multiple solutions to a problem are essentially the same, except for some permutation or rearrangement of values or variables



**Full symmetry:** any permutation preserves solutions. The full symmetry group $S_n$ has $n!$ symmetries over a sequence of $n$ elements

**Partial symmetry:** any piecewise permutation preserves solutions. This often occurs in instances. partial symmetry refers to a situation where only a subset of the variables or values in a problem exhibit symmetry.

**Wreath symmetry:** any wreath permutation preserves solutions. Wreath symmetry occurs when there are multiple groups of variables that exhibit symmetries independently, but these groups are connected in a hierarchical or structured manner.

**Rotation symmetry:** any rotation preserves solutions. The cyclic symmetry group $C_n$ has $n$ symmetries over a circular sequence of $n$ elements

**Index symmetry:** any permutation of slices of an array of decision variables preserves solutions.

**Conditional or dynamic symmetry:** a symmetry that appears while solving a problem. 



#### Propagator

A propagator for a predicate $\gamma$ removes from the current domains of the variables of a $\gamma$-constraint the values that cannot be part of a solution to that constraint.

- **Domain-consistency propagator** deletes all impossible values from the domains
- **Bounds-consistency propagator** only deletes all impossible min and max values from the domains

So in human language, a propagator helps guide the search process by enforcing constraints and pruning inconsistent values, leading to efficient and effective problem solving.



Each CP solver and LCG solver has a default propagator for each available constraint predicate. It is possible to override the defaults with annotations:

- `::domain ` asks for a domain-consistency propagator
- `::bounds` asks for a bounds-consistency propagator



#### Variable selection strategy

The variable selection strategy has an impact on the size of the search tree, especially of the constraints are processed with propagation at every node of the search tree or if the whole search tree is explored.

**First-fail principle**

To succeed: first try where you are most likely to fail.

In practice:

- Select a variable with the smallest current domain
- Select a variable involved in the largest #constraints
- Select a variable recently causing the most backtracks



**Best-first principle**

First try a domain part that is most likely, if not guaranteed, to have values that lead to solutions. This may be like how one would make the greedy choice in a greedy algorithm for the problem at hand.