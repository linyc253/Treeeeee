# Examine: Error varying with N and theta

In this testcase, we can examine the accuracy of treeeeee, computing the error between tree algorithm and brute force computation.

# How to use:
1. Run `sh run_all.sh` in the terminal in this folder.
2. Wait for the plot generated automatically.
3. When you no longer need the data, run `sh clean.sh` to clean up all the data.

# Composition of this testcase:
```
generate.py : set up the test initial condition
plot.py     : plot the accuracy curve
error.py    : compute the error by final data
run_all.sh  : the script written for the whole examine process
clean.sh    : the script cleans up all the data and figure.
```

# Parameter to be tuned:
No parameter to be tuned in this testcase.