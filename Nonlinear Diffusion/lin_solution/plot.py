import numpy as np
import matplotlib.pyplot as plt
import glob

# Adjust the file pattern as needed
file_pattern = "solution_N*_gamma*.dat"

# Find all solution files matching the pattern
files = glob.glob(file_pattern)

for filename in files:
    # Load the data (assuming two columns: x and u)
    data = np.loadtxt(filename)
    x = data[:, 0]
    u = data[:, 1]
    
    # Create a plot for this file
    plt.figure()
    plt.plot(x, u, 'b.-', label='u(x)')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title(f"Solution from {filename}")
    plt.legend()
    plt.grid(True)
    
    # Save the plot to a file (optional)
    plot_filename = filename.replace(".dat", "_plot.png")
    plt.savefig(plot_filename)
    plt.show()

