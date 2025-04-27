### You need to modify 'python' to your default python executable
PYTHON=python3

# Generate data
$PYTHON Initial_generator.py -N 10000 > Initial.dat

# Run METHOD = 2
    cat > Input_Parameter.ini<<!
[BasicSetting]
DIM = 3                      # Dimension of the system (default: 3)
METHOD = 2                   # Method 1:brute_force 
                             #        2:tree_algo (default)
PARTICLE_FILE = Initial.dat  # filename of particle file (default: Initial.dat)
N_TOT = 128                  # Number of base-level cell on each side
T_TOT = 10                  # total evolution time
DT = 0.1                     # maximal time interval
STEP_PER_OUT = 3

[FundamentalConst]
G   = 1                      # Newton's gravity constant

[Plummer]
N  = 20000                   # number of particles
a  = 200                    # critical length of the plummer
M  = 200                       # total mass of a plummer


[Tree]
THETA = 0.4                # Critical angle


[Plot]

!
../../bin/treeeeee

# Plot plummer animation
$PYTHON plot_gas.py