mkdir -p results

gcc Eigenvalue.c -o Eigenvalue
mpicc Eigenvalue_MPI.c -o Eigenvalue_MPI

FULL_MATRIX="bcsstk03.mtx"
mkdir -p submatrices
echo "" > results/convergence_matrix_size.txt

full_dim=$(grep -v '^%' $FULL_MATRIX | head -n 1 | awk '{print $1}')
for pct in 50 60 70 80 90 100
do
    new_size=$(( full_dim * pct / 100 ))
    submatrix_file="submatrices/bcsstk03_${new_size}.mtx"
    
    python reduce_matrix.py $FULL_MATRIX $submatrix_file $new_size
    
    output=$(mpirun -np 1 ./Eigenvalue_MPI $submatrix_file)
    echo "$output"
    iter=$(echo "$output" | grep "Iterations:" | awk '{print $11}')
    
    echo "$new_size $iter" >> results/convergence_matrix_size.txt
done

echo "" > results/convergence_process_count.txt
for np in 1 2 4 6 8
do
    output=$(mpirun -np $np ./Eigenvalue_MPI $FULL_MATRIX)
    echo "$output"
    iter=$(echo "$output" | grep "Iterations:" | awk '{print $11}')
    echo "$np $iter" >> results/convergence_process_count.txt
done

echo "Now plot !"
python3 ./plot.py

