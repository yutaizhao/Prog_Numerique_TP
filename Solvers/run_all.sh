output="output.txt"
# Complie the solver
gcc ex1.c -o solver
# Run the solver, save output to out.txt
./solver > $output
# Run the Python plotting script
python3 plot.py $output
