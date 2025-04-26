import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# load data named "Initial.dat"
file = "Initial.dat"
content = np.loadtxt(file)

print(content.size())