# Simulation: Interstellar Dust

In this testcase, we simulate random particles in the vacuum, considering them as the interstellar dust. We may see the collapse and explosion of the dust, forming a core.

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
plot_energy.py       : plot the energy curve
Figure/plot_gas.py   : visulize the data
run_all.sh           : the script written for the whole process from initialization to        
                       converting animation.
clean.sh             : the script cleans up all the data, figure, animation generated.
```

# Parameter to be tuned:
```
run_all.sh:
-N : Total particle number of the interstellar dust. [positive int]
-F : Number of frames for plotting. [positive int]

[BasicSetting]
DIM = 3                      # Dimension of the system (default: 3)
METHOD = 2                   # Method [1] brute_force, [2] tree_algo (default)
PARTICLE_FILE = Initial.dat  # filename of particle file (default: Initial.dat)
T_TOT = 100.0                # total evolution time
DT = 0.05                    # maximal time interval
ETA = 0.001                  # parameter that controls the accuracy and stability of the timestep in simulations
EPSILON = 1e-1               # softening length used to prevent singularities and numerical instabilities in particle interactions
TIME_PER_OUT = 0.5           # Output 00xxx.dat in every \Delta t = TIME_PER_OUT
OUTDIR = DATA                # where the output data stored
#RESTART = 12                # restart from 00012.dat

[Tree]
THETA = 0.4                  # Critical angle
POLES = 1                    # [1] dipole, [2] quadrupole
NCRIT = 1000                 # The max number of particles in a group (for constructing interaction list)

[Openmp]
THREADS = 4                  # Number of threads
CHUNK = 1                    # The chunk size in dynamic scheduling

[GPU]
threadsPerBlock = 128        # Slightly affect performance (try 32, 64, 128, 256, 512, 1024)
```

# Note:
If one find the total simulation time not long enough to see the full simulation you want, add `RESTART = 200` (change this number according to your last data file) in `Input_Parameter.ini, [BasicSetting]` to continue the following calculation with `00200.dat` as the new initial condition.