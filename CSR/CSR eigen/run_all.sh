#!/bin/bash
mkdir -p results

gcc CSR.c -o CSR
mpicc CSR_MPI.c -o CMPI
gcc Eigenvalue.c -o Eigenvalue
mpicc Eigenvalue_MPI.c -o Eigenvalue_MPI

# rm old results
rm -f results/CSR_serial.txt
for np in 2 4 6 8; do
  rm -f results/CSR_distr_${np}.txt
done

# run 30 times serial
echo "Running Serial CSR Implementation 30 times..."
for i in {1..30}; do
    echo "Run $i" >> results/CSR_serial.txt
    ./CSR >> results/CSR_serial.txt
done

# nb procs to test
procs=(2 4 6 8)
# run 30 times parallel
for p in "${procs[@]}"; do
  echo "Running Distributed CSR Implementation with ${p} processes 30 times..."
  for i in {1..30}; do
      echo "Run $i" >> results/CSR_distr_${p}.txt
      mpirun -np ${p} ./CMPI >> results/CSR_distr_${p}.txt
  done
done

echo "Experiments completed. Results are saved in the 'results' folder."

python3 ./plot.py

echo "..."
echo "Serial Power iteration method"
echo "..."
./Eigenvalue > "app_serial.txt"
echo "..."
echo "Distributed Power iteration method"
echo "..."
mpirun -np 4 ./Eigenvalue_MPI > "app_distr.txt"

