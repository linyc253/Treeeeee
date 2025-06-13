# Simulation: Double Plummer

In this testcase, we simulate two plummer merging. The two plummer are originally separated by a distance you set. You can set either the two plummer attract from still or rotating each other and finally collapse.

# How to use:
1. Tune your parameters (including all the parse in run_all.sh lines).
2. Run `sh run_all.sh` in the terminal in this folder.
3. Check `Initial.png` to see if the position of particle is satifying.
4. When treeeeee completes the calculation, check energy graphs.
5. Check the plotted figure is satisfying (the two plotting programs run simultaneously).
6. Check the final animation.
7. When you no longer need the data, run `sh clean.sh` to clean up all the data.

# Composition of this testcase:
```
Initial_generator.py : set up initial condition
plot.py              : visualize the initial condition
plot_energy.py       : plot the energy curve
Figure/plot_gas.py   : visulize the data (forward half frames)
Figure/plot_gas2.py  : visulize the data (latter half frames)
run_all.sh           : the script written for the whole process from initialization to        
                       converting animation.
clean.sh             : the script cleans up all the data, figure, animation generated.
```

# Parameter to be tuned:
```
run_all.sh:
-N : Total particle number of a single plummer. [positive int]
-V : Turn on/off the rotation (Defualt: 0). [0,1]
-F : Number of frames for plotting. [positive int]

[BasicSetting]
DIM = 3                      # Dimension of the system (default: 3)
METHOD = 2                   # Method [1]brute_force, [2]tree_algo (default)
PARTICLE_FILE = Initial.dat  # filename of particle file (default: Initial.dat)
T_TOT = 200                  # total evolution time
DT = 0.3                     # maximal time interval
ETA = 5.0                    # parameter that controls the adaptive timestep
TIME_PER_OUT = 0.5           # Output 00xxx.dat in every STEP_PER_OUT steps
EPSILON = 1e-5               # softening length
OUTDIR = DATA                # the direction of output data

[Tree]
THETA = 0.4                  # Critical angle
POLES = 1                    # dipole [1], quadrupole[2]
NCRIT = 1000                 # The max number of particles in a group

[Openmp]
THREADS = 4                  # Number of threads
CHUNK = 1                    # The chunk size in dynamic scheduling
```
Also, don't forget:
```
Initial_generator.py:
a = 20                      # critical radius
d = 200                     # distance between two plummer
M = 1000                    # total mass
```
# Note:
If one find the total simulation time not long enough to see the full simulation you want, add `RESTART = 400` (change this number according to your last data file) in `Input_Parameter.ini, [BasicSetting]` to continue the following calculation with `00400.dat` as the new initial condition.