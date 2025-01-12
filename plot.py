import re
import matplotlib.pyplot as plt

jacobi_iters = []
jacobi_resid = []
gs_iters = []
gs_resid = []

with open("out.txt", "r") as f:
    for line in f:
        jacobi_match = re.match(r"Jacobi iter\s+(\d+):\s+relative residual = ([0-9.eE+-]+)", line)
        if jacobi_match:
            k = int(jacobi_match.group(1))
            val = float(jacobi_match.group(2))
            jacobi_iters.append(k)
            jacobi_resid.append(val)
            continue
        
        gs_match = re.match(r"Gauss-Seidel iter\s+(\d+):\s+relative residual = ([0-9.eE+-]+)", line)
        if gs_match:
            k = int(gs_match.group(1))
            val = float(gs_match.group(2))
            gs_iters.append(k)
            gs_resid.append(val)
            continue

plt.figure()

plt.plot(jacobi_iters, jacobi_resid, marker='o', label='Jacobi')
plt.plot(gs_iters, gs_resid, marker='s', label='Gauss-Seidel')

plt.xlabel('#Iteration')
plt.ylabel('Relative Residual')
plt.title('Convergence of 2 Methods')
plt.grid(True)
plt.legend()
plt.show()
