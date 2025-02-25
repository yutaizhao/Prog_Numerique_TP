mpicc -I/Users/zhaoyutainazir/Library/hypre/src/hypre/include \
      -L/Users/zhaoyutainazir/Library/hypre/src/hypre/lib -lHYPRE \
      -o equation equation.c

mkdir -p results/implicit
mkdir -p results/newton

echo "Running Linearized Implicit (method=1)..."
mpirun -np 1 ./equation 1 1 > results/implicit/implicit_set1.txt
mpirun -np 1 ./equation 1 2 > results/implicit/implicit_set2.txt

echo "Running Newton (method=2)..."
mpirun -np 1 ./equation 2 1 > results/newton/newton_set1.txt
mpirun -np 1 ./equation 2 2 > results/newton/newton_set2.txt

echo "Execution finished. Now plot !"
