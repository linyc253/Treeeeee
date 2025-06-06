import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# ----------------------
# (1) Parse arguments
# ----------------------
parser = argparse.ArgumentParser()
# parser.add_argument("-H", type=int, default=0,
#     help="Turn on/off the halo. Default: 0.")
parser.add_argument("-R", type=float, default=0.8,
    help="Bulge/disk mass‐distribution ratio. Default: 0.8.")
args = parser.parse_args()
bd_rate = args.R
k = int(1 + 5 * bd_rate)

# ----------------------
# (2) Read in the data
# ----------------------
filename1 = "S0H1.dat"
filename2 = "S0H0.dat"
with open(filename1, 'r') as f:
    lines1 = f.readlines()
with open(filename2, 'r') as f:
    lines2 = f.readlines()

Npt1 = int(lines1[0].strip())
Npt2 = int(lines2[0].strip())
Npt1 = int(Npt1 / k)

# --- Read positions into an (N × 3) array r_raw ---
r_raw1 = np.array([
        list(map(float, lines1[i].split()))
        for i in range(1 + k * Npt1, 1 + (k + 1) * Npt1)
])
r_raw2 = np.array([
    list(map(float, lines2[i].split()))
    for i in range(1 + Npt2, 1 + 2 * Npt2)
])

# Compute |r| for each particle
r_abs1 = np.sqrt((r_raw1**2).sum(axis=1))    # shape = (Npt,)
r_abs2 = np.sqrt((r_raw2**2).sum(axis=1))

# --- Read velocities into an (N × 3) array v_raw ---
v_raw1 = np.array([
    list(map(float, lines1[i].split()))
    for i in range(1 + 2 * k * Npt1, 1 + (2 * k + 1) * Npt1)
])
v_raw2 = np.array([
    list(map(float, lines2[i].split()))
    for i in range(1 + 2 * Npt2, 1 + 3 * Npt2)
])

# ------------------------------
# (3) Compute tangent speeds v_tan
# ------------------------------
#
# Formula: v_t = v_raw - [(v_raw ⋅ r_raw)/(r_raw ⋅ r_raw)] * r_raw, 
# and v_tan[i] = ||v_t[i]|| for each i.
#
# We can do this in a vectorized way:

# Compute dot(v_raw, r_raw) and ||r_raw||^2 for each particle:
proj_factor1 = (v_raw1 * r_raw1).sum(axis=1)               # shape = (N,)
r_norm21     = (r_raw1 * r_raw1).sum(axis=1)              # shape = (N,)
proj_factor2 = (v_raw2 * r_raw2).sum(axis=1)
r_norm22     = (r_raw2 * r_raw2).sum(axis=1)

# Avoid any division‐by‐zero (if a particle happened to be exactly at r = 0).
# In practice r = 0 is extremely unlikely, but let’s be safe:
r_norm2_safe1 = np.where(r_norm21 == 0, 1e-16, r_norm21)
r_norm2_safe2 = np.where(r_norm22 == 0, 1e-16, r_norm22)

# Compute the projected component (scalar) / r_norm2 for each particle:
scalars1 = proj_factor1 / r_norm2_safe1                    # shape = (N,)
scalars2 = proj_factor2 / r_norm2_safe2

# Now subtract that projection from v_raw to get v_t (tangential component):
v_t1 = v_raw1 - (scalars1[:, None] * r_raw1)               # shape = (N,3)
v_t2 = v_raw2 - (scalars2[:, None] * r_raw2)

# Finally, the tangent speed is the Euclidean length of v_t:
v_tan1 = np.sqrt((v_t1 * v_t1).sum(axis=1))               # shape = (N,)
v_tan2 = np.sqrt((v_t2 * v_t2).sum(axis=1))

# -------------------------------------------------------
# (4) Bin those v_tan values by integer radius [0..6999]
# -------------------------------------------------------
#
# We want to group every particle whose radius satisfies n ≤ r < n+1 
# into “bin n”, for n = 0 … 6999 (so that the x‐axis of our final plot 
# runs from 0 to 6999). Then we average the tangent speeds in each bin.

max_bin = 7000
#  (any particle with r_abs >= 7000 we will simply ignore, since xlim=7000)

# Compute integer bin index for each particle:
bin_index1 = r_abs1.astype(int)   # floor(r_abs).  E.g. 3.14 -> bin 3
bin_index2 = r_abs2.astype(int)

# Discard all particles whose bin_index >= max_bin
valid_mask1 = (bin_index1 >= 0) & (bin_index1 < max_bin)
bin_index1 = bin_index1[valid_mask1]
v_tan_valid1 = v_tan1[valid_mask1]

valid_mask2 = (bin_index2 >= 0) & (bin_index2 < max_bin)
bin_index2 = bin_index2[valid_mask2]
v_tan_valid2 = v_tan2[valid_mask2]

# Now use np.bincount to sum up v_tan in each bin, and count how many per bin:
sum_per_bin1   = np.bincount(bin_index1, weights=v_tan_valid1, minlength=max_bin)
sum_per_bin2   = np.bincount(bin_index2, weights=v_tan_valid2, minlength=max_bin)
count_per_bin1 = np.bincount(bin_index1, minlength=max_bin)
count_per_bin2 = np.bincount(bin_index2, minlength=max_bin)

# Compute the average tangential speed in each bin; 
# wherever count_per_bin == 0, set avg to np.nan so it won’t get plotted.
with np.errstate(divide='ignore', invalid='ignore'):
    avg_v_tan_per_bin1 = sum_per_bin1 / count_per_bin1
    avg_v_tan_per_bin2 = sum_per_bin2 / count_per_bin2
    avg_v_tan_per_bin1[count_per_bin1 == 0] = np.nan
    avg_v_tan_per_bin2[count_per_bin2 == 0] = np.nan

# Now avg_v_tan_per_bin is a length‐7000 array, where
#    avg_v_tan_per_bin[n] = mean v_tan of all particles with n ≤ r < n+1,
# or np.nan if no particles fell into that bin.

# --------------------------------
# (5) Make the “clean” final plot
# --------------------------------
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot()

# Plot the averaged curve vs. bin index. 
#   x = 0,1,2,…,6999  and  y = avg_v_tan_per_bin[x].
x_bins = np.arange(max_bin)

ax.scatter(x_bins, avg_v_tan_per_bin1, color='b', s=0.1, label='halo(blue)')
ax.scatter(x_bins, avg_v_tan_per_bin2, color='g', s=0.1, label='non-halo(green)')
ax.set_xlim(0, 7000)
ax.set_ylim(0, 2.7)
ax.set_xlabel("Radius")
ax.set_ylabel("Velocity")
ax.set_title("Galaxy Velocity Curve", fontsize=25)
ax.legend(fontsize=14)

plt.tight_layout()
fig.savefig("velocity_curve_binned.png")
plt.close()
