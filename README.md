# Quick Start
First, clone using the following command
```
git clone https://github.com/linyc253/Treeeeee.git
cd Treeeeee
```
## AutoTool Configure
To compile the code, first configure by
```
./configure.sh --enable-gpu --enable-openmp
```
Depending on your system, you might need to specify the CUDA directory manually `--cuda-home=/your_path_to_cuda/cuda` in the command above. Then, build the program by
```
make
```
(whenever you modify the compiler option, please do `make clean` before `make`)

## CMake
Alternatively, you can also build through CMake by executing the following
```
cmake -S . -B build && cmake --build build
```
Additional cmake options can be turned on, `OMP` for OpenMP, `CUDA` for cuda support, and `DEBUG` for some additional timing information. For example, 
```
cmake -DOMP=ON -DCUDA=ON -S . -B build && cmake --build build
```

# How to Run
Once compiled you can test by executing the following to run a test case.
```
cd testcase/test_random
python Initial_generator.py -N 10000 > Initial.dat
../../bin/treeeeee
```
You can check whether the energy conservation by plotting,
```
python plot_energy.py
```
and the calculation result can be visualized via python scripts. 
```
cd Figure
python plot_gas.py -F 200
cd ..
```
Make sure modules in "requirements.txt" are installed. The figures can be converted to GIF
```
convert Figure/*.png random.gif
```
or MP4 movie
```
ffmpeg -framerate 12 -pix_fmt yuv420p -i Figure/%05d.png random.mp4
```

# Parameter file format
The parameter file should be named as `Input_Parameter.ini`. Use `#` for comments.
```
[BasicSetting]
DIM = 3                      # Dimension of the system (default: 3)
METHOD = 2                   # Method [1] brute_force, [2] tree_algo (default)
PARTICLE_FILE = Initial.dat  # filename of particle file (default: Initial.dat)
T_TOT = 100.0                # total evolution time
DT = 0.05                    # maximal time interval
ETA = 0.001                  # parameter that controls the accuracy and stability of the timestep in simulations
EPSILON = 1e-1               # softening length used to prevent singularities and numerical instabilities in particle interactions
TIME_PER_OUT = 0.5           # Output 00xxx.dat in every \Delta t = TIME_PER_OUT
OUTDIR = data                # where the output data stored
# RESTART = 12               # restart from 00012.dat

[Tree]
THETA = 0.4                  # Critical angle
POLES = 1                    # [1] dipole, [2] quadrupole
NCRIT = 1000                 # The max number of particles in a group (for generating interaction list)

[Openmp]
THREADS = 4                  # Number of threads
CHUNK = 1                    # The chunk size in dynamic scheduling

[GPU]
threadsPerBlock = 128        # Slightly affect performance (try 32, 64, 128, 256, 512, 1024)
```
# Particle file format
The particle file (filename should be specified in `Input_Parameter.ini`) must follow the format below
```
N                             # number of particles
mass(1)                       # ---| 
...                           #    | N lines of masses
mass(n_particles)             # ___|
x(1),y(1),z(1)                # ------| 
...                           #       | N lines of initial particle positions
x(N),y(N),z(N)                # ______|
vx(1),vy(1),vz(1)             # ---------| 
...                           #          | N lines of initial velocities
vx(N),vy(N),vz(N)             # _________|
```
