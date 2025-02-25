import re
import matplotlib.pyplot as plt

def extract_times(filename, label):
    times = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                if label in line:
                    m = re.search(r'([0-9]*\.?[0-9]+)', line)
                    if m:
                        times.append(float(m.group(1)))
    except FileNotFoundError:
        print(f"Warning: File {filename} not found!")
    return times

# Extract serial execution time
serial_times = extract_times("results/CSR_serial.txt", "Serial matvec time:")
if not serial_times:
    print("No serial timing data found in results/CSR_serial.txt!")
    serial_times = [1.0]  # Placeholder to avoid division by zero
mean_serial = sum(serial_times) / len(serial_times)
print("Mean Serial Time: {:.6f} sec".format(mean_serial))

# Define process counts to test
procs = [2, 4, 6, 8]
mean_distr = {}
speedups = {}
efficiencies = {}

# Extract distributed execution times for different process counts
for p in procs:
    filename = f"results/CSR_distr_{p}.txt"
    times = extract_times(filename, "Average matvec time:")
    if not times:
        print(f"No distributed timing data found for {p} processes in {filename}!")
        continue
    mean_time = sum(times) / len(times)
    mean_distr[p] = mean_time
    speedups[p] = mean_serial / mean_time
    efficiencies[p] = mean_serial / (mean_time * p)
    print(f"Processes: {p}, Mean Distributed Time: {mean_time:.6f} sec, Speedup: {speedups[p]:.2f}, Efficiency: {efficiencies[p]:.2f}")

# Ensure we have data for all processes before plotting
if any(p not in mean_distr for p in procs):
    print("Missing distributed time data for some process counts. Check the output files.")

# Plot first figure: Mean Execution Time for Serial and Distributed Implementations
plt.figure(figsize=(8, 5))
labels = ['Serial'] + [f'Distributed (np={p})' for p in procs]
times_bar = [mean_serial] + [mean_distr.get(p, 0) for p in procs]
plt.bar(labels, times_bar, color=['blue'] + ['green'] * len(procs))
plt.ylabel("Mean Matvec Time (sec)")
plt.title("Mean Execution Time (Serial vs Distributed)")
plt.xticks(rotation=45)  # Rotate x-axis labels for clarity
for i, t in enumerate(times_bar):
    plt.text(i, t, f"{t:.6f}", ha='center', va='bottom')
plt.tight_layout()
plt.savefig("results/mean_execution_time.png")
plt.show()

# Plot second figure: Speedup & Efficiency in a subplot
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Speedup (Serial Time / Distributed Time)
labels_proc = [str(p) for p in procs]
speedup_values = [speedups.get(p, 0) for p in procs]
axs[0].bar(labels_proc, speedup_values, color='orange')
axs[0].set_xlabel("Number of Processes")
axs[0].set_ylabel("Speedup")
axs[0].set_title("Speedup of Distributed Implementation")
for i, s in enumerate(speedup_values):
    axs[0].text(i, s, f"{s:.2f}", ha='center', va='bottom')

# Efficiency (Speedup / Number of Processes)
efficiency_values = [efficiencies.get(p, 0) for p in procs]
axs[1].bar(labels_proc, efficiency_values, color='red')
axs[1].set_xlabel("Number of Processes")
axs[1].set_ylabel("Efficiency")
axs[1].set_title("Efficiency of Distributed Implementation")
for i, e in enumerate(efficiency_values):
    axs[1].text(i, e, f"{e:.2f}", ha='center', va='bottom')

plt.tight_layout()
plt.savefig("results/performance_matvec.png")
plt.show()
