import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Change this
data_dir = "data4/"  # folder with your files




file_pattern = "{:05d}.dat"  # adjust to match your filenames
num_frames = 300  # adjust to the number of files you have

def read_particle_positions(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    # Extract particles starting from line 4
    coords = []
    npart = int(lines[0])
    for i in range(npart):
        parts = lines[i + npart + 1].strip().split()
        x, y = float(parts[0]), float(parts[1])
        coords.append((x, y))
    return coords

# Read all frames
frames_data = []
for i in range(num_frames):
    filepath = os.path.join(data_dir, file_pattern.format(i + 1))
    if os.path.exists(filepath):
        frames_data.append(read_particle_positions(filepath))

# Plotting
fig, ax = plt.subplots()
scat = ax.scatter([], [])

def init():
    ax.set_xlim(-10, 10)  # Set based on your coordinate range
    ax.set_ylim(-10, 10)
    return scat,

def update(frame_data):
    x_vals, y_vals = zip(*frame_data)
    scat.set_offsets(list(zip(x_vals, y_vals)))
    return scat,

ani = animation.FuncAnimation(
    fig, update, frames=frames_data, init_func=init, blit=True, interval=100
)

plt.show()
