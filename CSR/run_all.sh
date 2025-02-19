# Compilation
gcc CSR.c -o CSR
mpicc CSR_MPI.c -o CMPI
gcc Eigenvalue.c -o Eigenvalue
mpicc Eigenvalue_MPI.c -o Eigenvalue_MPI
# Run
echo "..."
echo "Serial CSR Implementation"
echo "..."
./CSR
echo "..."
echo "Distributed CSR Implementation"
echo "..."
mpirun -np 4 ./CMPI
echo "..."
echo "Serial Power iteration method"
echo "..."
./Eigenvalue
echo "..."
echo "Distributed Power iteration method"
echo "..."
mpirun -np 4 ./Eigenvalue_MPI

