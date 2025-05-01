### You need to modify 'python' to your default python executable
PYTHON=/home/linyc253/.conda/envs/env_1/bin/python

# Generate data
cat > Initial.dat<<!
3
5.0
4.0
3.0
3.0  0.0 0.0
-3.0  0.0 0.0
0.0 1.0 0.0
0.0  0.4 0.0
0.0 -0.5 0.0
0.0 0.0 0.0
!

# Run 1
cat > Input_Parameter.ini<<!
[BasicSetting]
DIM = 3                      # Dimension of the system (default: 3)
METHOD = 2                   # Method 1:brute_force 
                             #        2:tree_algo (default)
PARTICLE_FILE = Initial.dat  # filename of particle file (default: Initial.dat)
T_TOT = 30.0                 # total evolution time
DT = 0.01                    # maximal time interval
ETA = 1.0                    # parameter that controls the accuracy and stability of the timestep in simulations
EPSILON = 1e-4               # softening length used to prevent singularities and numerical instabilities in particle interactions
TIME_PER_OUT = 0.1           # Output 00xxx.dat in every STEP_PER_OUT steps
OUTDIR = data1               # directory to store the data (default: "." [the execution directory])

[Tree]
THETA = 0.5                  # Critical angle
POLES = 1                    # 1: dipole (centre of mass)
                             # 2: quadrupole (3 pseudo-particles)
!
../../bin/treeeeee > log1
cd data1
$PYTHON ../Energy.py
mv Energy.png ../Energy1.png
cd ..

# Run 2 (larger EPSILON may also cause something strange!!)
cat > Input_Parameter.ini<<!
[BasicSetting]
DIM = 3                      # Dimension of the system (default: 3)
METHOD = 2                   # Method 1:brute_force 
                             #        2:tree_algo (default)
PARTICLE_FILE = Initial.dat  # filename of particle file (default: Initial.dat)
T_TOT = 30.0                 # total evolution time
DT = 0.01                    # maximal time interval
ETA = 1.0                    # parameter that controls the accuracy and stability of the timestep in simulations
EPSILON = 1e-3               # softening length used to prevent singularities and numerical instabilities in particle interactions
TIME_PER_OUT = 0.1           # Output 00xxx.dat in every STEP_PER_OUT steps
OUTDIR = data2               # directory to store the data (default: "." [the execution directory])

[Tree]
THETA = 0.5                  # Critical angle
POLES = 1                    # 1: dipole (centre of mass)
                             # 2: quadrupole (3 pseudo-particles)
!
../../bin/treeeeee > log2
cd data2
$PYTHON ../Energy.py
mv Energy.png ../Energy2.png
cd ..


# Run 3 (disable adaptive time step by comment out ETA)
cat > Input_Parameter.ini<<!
[BasicSetting]
DIM = 3                      # Dimension of the system (default: 3)
METHOD = 2                   # Method 1:brute_force 
                             #        2:tree_algo (default)
PARTICLE_FILE = Initial.dat  # filename of particle file (default: Initial.dat)
T_TOT = 30.0                 # total evolution time
DT = 0.01                    # maximal time interval
#ETA = 1.0                    # parameter that controls the accuracy and stability of the timestep in simulations
EPSILON = 1e-4               # softening length used to prevent singularities and numerical instabilities in particle interactions
TIME_PER_OUT = 0.1           # Output 00xxx.dat in every STEP_PER_OUT steps
OUTDIR = data3               # directory to store the data (default: "." [the execution directory])

[Tree]
THETA = 0.5                  # Critical angle
POLES = 1                    # 1: dipole (centre of mass)
                             # 2: quadrupole (3 pseudo-particles)
!
../../bin/treeeeee > log3
cd data3
$PYTHON ../Energy.py
mv Energy.png ../Energy3.png
cd ..

# Run 4 (disable adaptive time step by comment out ETA, decrease DT)
cat > Input_Parameter.ini<<!
[BasicSetting]
DIM = 3                      # Dimension of the system (default: 3)
METHOD = 2                   # Method 1:brute_force 
                             #        2:tree_algo (default)
PARTICLE_FILE = Initial.dat  # filename of particle file (default: Initial.dat)
T_TOT = 30.0                 # total evolution time
DT = 0.001                    # maximal time interval
#ETA = 1.0                    # parameter that controls the accuracy and stability of the timestep in simulations
EPSILON = 1e-4               # softening length used to prevent singularities and numerical instabilities in particle interactions
TIME_PER_OUT = 0.1           # Output 00xxx.dat in every STEP_PER_OUT steps
OUTDIR = data4               # directory to store the data (default: "." [the execution directory])

[Tree]
THETA = 0.5                  # Critical angle
POLES = 1                    # 1: dipole (centre of mass)
                             # 2: quadrupole (3 pseudo-particles)
!
../../bin/treeeeee > log4
cd data4
$PYTHON ../Energy.py
mv Energy.png ../Energy4.png
cd ..


echo "Use plot.py to see what happen, and use 'tail log1' to see how many iteration steps are cost."
echo "You should find that the adaptive time step is advantageous in this case."