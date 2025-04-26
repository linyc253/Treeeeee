import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Assume the data is with the form: m x[i] y[i] z[i] vx[i] vy[i] vz[i]

# load final data named "Data_0000.dat"
file = "Data_0000.dat"
content = np.loadtxt(file)
