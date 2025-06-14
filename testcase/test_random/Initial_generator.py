import random
import argparse

def generate_nbody_testcase(npart, mass_range=(0.1, 0.4), pos_range=(-100.0, 100.0), vel_range=(-1.5, 1.5)):
    masses = [random.uniform(*mass_range) for _ in range(npart)]
    positions = [(random.uniform(*pos_range), random.uniform(*pos_range), random.uniform(*pos_range)) for _ in range(npart)]
    velocities = [(random.uniform(*vel_range), random.uniform(*vel_range), random.uniform(*vel_range)) for _ in range(npart)]
    
    print(npart)
    
    for m in masses:
        print(f"{m:.6f}")
        
    for x, y, z in positions:
        print(f"{x:.6f} {y:.6f} {z:.6f}")
        
    for vx, vy, vz in velocities:
        print(f"{vx:.6f} {vy:.6f} {vz:.6f}")

if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--npart", help = "Number of particles")
    args = parser.parse_args()

    # Generate testcase
    generate_nbody_testcase(npart=int(args.npart))
