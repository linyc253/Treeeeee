### You need to modify 'python' to your default python executable
PYTHON=python

# Generate data
$PYTHON Initial_generator.py -N 10000 > Initial.dat

### Run METHOD = 2
    cat > Input_Parameter.ini<<!
[BasicSetting]
DIM = 3                      # Dimension of the system (default: 3)
METHOD = 2                   # Method [1]brute_force, [2]tree_algo (default)
PARTICLE_FILE = Initial.dat  # filename of particle file (default: Initial.dat)
T_TOT = 100.0                # total evolution time
DT = 0.05                    # maximal time interval
ETA = 0.05                   # parameter that controls the adaptive timestep
TIME_PER_OUT = 0.5           # Output 00xxx.dat in every STEP_PER_OUT steps
EPSILON = 1e-3               # softening length
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

# Plot animation
$PYTHON plot_energy.py

cd Figure
$PYTHON plot_gas.py -F 200
cd ..
ffmpeg -framerate 12 -i Figure/%05d.png random.mp4