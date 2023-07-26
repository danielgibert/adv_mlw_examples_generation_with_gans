# Approximated approach: Getting the solution with a constraints solver

The set of constraints are modeled in zimpl format, in order to be
easy to be read. Then, `zimpl` will translate the problem to a
constraint solver format.

First, the zpl file needs to be translated to the lp format (the default format):

     $ zimpl padding.zpl

It creates two files: padding.lp and padding.tbl. The first is
the set of constraints, the second is a table translating the
variable names in zpl to the ones in lp.

Finally, to get the solution, the solver soplex is called:

    $ soplex padding.lp

Zimpl and soplex are included in the `scip` package, which can be downloaded from https://www.scipopt.org/index.php#download 
