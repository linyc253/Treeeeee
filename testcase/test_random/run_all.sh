### You need to modify 'python' to your default python executable
PYTHON=/home/linyc253/.conda/envs/env_1/bin/python

# Generate data
$PYTHON generate.py -N 10000 > Initial.dat

../../bin/treeeeee

# Plot plummer animation
$PYTHON plot_gas.py
convert data/*.png Random.gif