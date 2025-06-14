### You need to modify 'python' to your default python executable
PYTHON=python

### Generate data (modfy N below)
$PYTHON Initial_generator.py > Initial.dat
#$PYTHON random_generator.py -N 10 > Initial.dat

### Setting
    cat > Input_Parameter.ini<<!
[BasicSetting]
DIM = 3                      # Dimension of the system (default: 3)
METHOD = 1                   # Method 1:brute_force 
                             #        2:tree_algo (default)
PARTICLE_FILE = Initial.dat  # filename of particle file (default: Initial.dat)
T_TOT = 10                   # total evolution time
DT = 0.02                    # maximal time interval
ETA = 1.0                    # parameter that controls the accuracy and stability of the timestep in simulations
TIME_PER_OUT = 0.05          # Output 00xxx.dat in every STEP_PER_OUT steps
EPSILON = 1e-4               # softening length used to prevent singularities and numerical instabilities in particle interactions
OUTDIR = DATA

[Tree]
THETA = 0.4                  # Critical angle
POLES = 1
NCRIT = 1000

[Openmp]
THREADS = 4
CHUNK = 1
!
../../bin/treeeeee > log

### Plot plummer animation
$PYTHON plot_energy.py
cd Figure
$PYTHON plot.py -F 200
cd ..
ffmpeg -framerate 12 -i Figure/%05d.png solar_sys.mp4