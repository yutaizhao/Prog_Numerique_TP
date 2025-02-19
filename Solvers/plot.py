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
cg_iters = []
cg_resid = []
gmres_iters = []
gmres_resid = []

# Open the specified file
with open(args.file, "r") as f:
    for line in f:
        
        # Match Conjugate Gradient iteration lines
        cg_match = re.match(r"Conjugate Gradient\s+(\d+):\s+relative residual = ([0-9.eE+-]+)", line)
        if cg_match:
            k = int(cg_match.group(1))
            val = float(cg_match.group(2))
            cg_iters.append(k)
            cg_resid.append(val)
            continue
        
        # Match GMRES iteration lines
        gmres_match = re.match(r"GMRES\s+(\d+):\s+relative residual = ([0-9.eE+-]+)", line)
        if gmres_match:
            k = int(gmres_match.group(1))
            val = float(gmres_match.group(2))
            gmres_iters.append(k)
            gmres_resid.append(val)
            continue
            
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

# Plot each method
plt.plot(cg_iters, cg_resid, marker='^', label='Conjugate Gradient')
plt.plot(gmres_iters, gmres_resid, marker='x', label='GMRES')
plt.plot(jacobi_iters, jacobi_resid, marker='o', label='Jacobi')
plt.plot(gs_iters, gs_resid, marker='s', label='Gauss-Seidel')


# Add labels, grid, and legend
plt.xlabel('#Iteration')
plt.ylabel('Relative Residual')
plt.title('Convergence of Iterative Methods')
plt.yscale('log')  # Use logarithmic scale for residuals
plt.grid(True)
plt.legend()

# Save the plot
plt.savefig("./convergence_plot.png")
plt.show()

