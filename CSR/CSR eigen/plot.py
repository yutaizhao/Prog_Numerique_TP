import matplotlib.pyplot as plt

sizes = []
iterations_size = []
with open('results/convergence_matrix_size.txt', 'r') as f:
    for line in f:
        if line.strip():
            size, iters = map(float, line.strip().split())
            sizes.append(size)
            iterations_size.append(iters)

processes = []
iterations_proc = []
with open('results/convergence_process_count.txt', 'r') as f:
    for line in f:
        if line.strip():
            proc, iters = map(float, line.strip().split())
            processes.append(proc)
            iterations_proc.append(iters)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(sizes, iterations_size, marker='o')
ax1.set_xlabel('Matrix Size (n)')
ax1.set_ylabel('Number of Iterations')
ax1.set_title('Convergence vs Matrix Size')
ax1.grid(True)

ax2.plot(processes, iterations_proc, marker='o')
ax2.set_xlabel('Number of Processes')
ax2.set_ylabel('Number of Iterations')
ax2.set_title('Convergence vs Number of Processes')
ax2.grid(True)

plt.tight_layout()
plt.savefig('results/convergence_csr_eigen.png')
plt.show()
