# Compilation
gcc CSR.c -o CSR
mpicc CSR_MPI.c -o CMPI
# Run
echo "..."
echo "Serial CSR Implementation"
echo "..."
./CSR
echo "..."
echo "Distributed CSR Implementation"
echo "..."
mpirun -np 4 ./CMPI

