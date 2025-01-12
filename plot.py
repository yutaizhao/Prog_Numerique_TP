import re
import argparse
import matplotlib.pyplot as plt

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Plot convergence from output.")
parser.add_argument("file", type=str, help="Path to the output file.")
args = parser.parse_args()

# Initialize lists for iterations and residuals
jacobi_iters = []
jacobi_resid = []
gs_iters = []
gs_resid = []

# Open the specified file
with open(args.file, "r") as f:
    for line in f:
        # Match Jacobi iteration lines
        jacobi_match = re.match(r"Jacobi\s+(\d+):\s+relative residual = ([0-9.eE+-]+)", line)
        if jacobi_match:
            k = int(jacobi_match.group(1))
            val = float(jacobi_match.group(2))
            jacobi_iters.append(k)
            jacobi_resid.append(val)
            continue
        
        # Match Gauss-Seidel iteration lines
        gs_match = re.match(r"Gauss-Seidel\s+(\d+):\s+relative residual = ([0-9.eE+-]+)", line)
        if gs_match:
            k = int(gs_match.group(1))
            val = float(gs_match.group(2))
            gs_iters.append(k)
            gs_resid.append(val)
            continue

# Plot the data
plt.figure()

plt.plot(jacobi_iters, jacobi_resid, marker='o', label='Jacobi')
plt.plot(gs_iters, gs_resid, marker='s', label='Gauss-Seidel')

plt.xlabel('#Iteration')
plt.ylabel('Relative Residual')
plt.title('Convergence of 2 Methods')
plt.grid(True)
plt.legend()
plt.savefig("./plot.png")
