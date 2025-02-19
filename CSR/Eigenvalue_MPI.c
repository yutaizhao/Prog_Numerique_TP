#include <mpi.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//Contruct CSR matrix from a mtx file
void constructCSR(char *filename, int **csr_row_ptr, int **csr_col_idx,
                  double **csr_vals, int *csr_n, int *csr_num_values)
{
    /** Preprocessing **/
    
    FILE *f = fopen(filename, "r");
    if (!f) {exit(1);}
    
    // Skip '%'
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] != '%')
            break;
    }
    // Read matrix size and number of values (first row) after commentaires
    int M, N, nb_values;
    if (sscanf(line, "%d %d %d", &M, &N, &nb_values) != 3) {exit(1);}
    if (M != N) {exit(1);}
    
    /** Lecture **/
    
    // Read file in COO format
    int max_size = 2 * nb_values; //Estime total values of the matrix
    int *coo_rows = (int *)malloc(max_size * sizeof(int));
    int *coo_cols = (int *)malloc(max_size * sizeof(int));
    double *coo_vals = (double *)malloc(max_size * sizeof(double));
    
    int count = 0;
    int r, c;
    double val;
    
    for (int i = 0; i < nb_values; i++) {
        if (fscanf(f, "%d %d %lf", &r, &c, &val) != 3) {exit(1);}
        // Since index started with 1 in dataset
        r--;
        c--;
        // Save (r, c, val)
        coo_rows[count] = r;
        coo_cols[count] = c;
        coo_vals[count] = val;
        count++;
        // For non diagonal terms (c, r, val) add the symetric values
        if (r != c) {
            coo_rows[count] = c;
            coo_cols[count] = r;
            coo_vals[count] = val;
            count++;
        }
    }
    
    fclose(f);
    int total_size = count;  // Real total values of the matrix
    
    /** Construction of CSR  **/
    
    int *row_ptr = (int *)calloc((M+1), sizeof(int));
    int *col_idx = (int *)malloc(total_size * sizeof(int));
    double *vals = (double *)malloc(total_size * sizeof(double));
    
    // Number of values in each row
    for (int i = 0; i < total_size; i++) {
        row_ptr[coo_rows[i] + 1]++;
    }
    // However row_ptr[i] = Cumulative number of values in each row
    for (int i = 0; i < M; i++) {
        row_ptr[i+1] += row_ptr[i];
    }
    
    // Copie de row_ptr
    int *temp = (int *)malloc(M * sizeof(int));
    for (int i = 0; i < M; i++) {
        temp[i] = row_ptr[i];
    }
    
    // Transfer COO data into CSR
    for (int i = 0; i < total_size; i++) {
        int row = coo_rows[i];
        int pos = temp[row]++; //temp[row] = Cumulative number of values in the row, +1 : next value
        col_idx[pos] = coo_cols[i];
        vals[pos] = coo_vals[i];
    }
    
    free(temp);
    free(coo_rows);
    free(coo_cols);
    free(coo_vals);
    
    *csr_row_ptr = row_ptr;
    *csr_col_idx = col_idx;
    *csr_vals = vals;
    *csr_n = M;
    *csr_num_values = total_size;
}

// construct a vector
double* constructVector(int n)
{
    double *vec = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        vec[i] = 1;
    }
    return vec;
}


//y = A*x, A in CSR format
void csr_matvec(const int n, const int *row_ptr, const int *col_idx,
                const double *vals, const double *x,
                double *y)
{
    // n = number of rows
    for(int i = 0; i < n; i++){
        double row_sum = 0.0;
        int start = row_ptr[i];
        int end   = row_ptr[i+1];
        for(int k = start; k < end; k++) {
            row_sum += vals[k] * x[col_idx[k]];
        }
        y[i] = row_sum;
    }
}

double distributed_power_iteration(double *v, int local_n, int global_n,
                                   const int *local_row_ptr, const int *local_col_idx, const double *local_vals,
                                   double tol, int max_iter, int *iter_count)
{
    double alpha_prev = 0.0, alpha = 0.0;
    double *local_y = (double *)malloc(local_n * sizeof(double));
    double *global_y = (double *)malloc(global_n * sizeof(double));
    
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // rank 0 gets: local_n0 = local_n_init + remainder, others get: local_n_init
    int local_n_init = global_n / size;
    int remainder = global_n % size;
    
    int k;
    for (k = 0; k < max_iter; k++) {
        // Compute local_y = A_local * v
        csr_matvec(local_n, local_row_ptr, local_col_idx, local_vals, v, local_y);
        
        // Compute local maximum absolute value
        double local_max = 0.0;
        for (int i = 0; i < local_n; i++) {
            double abs_val = fabs(local_y[i]);
            if (abs_val > local_max)
                local_max = abs_val;
        }
        // Get global maximum
        MPI_Allreduce(&local_max, &alpha, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        
        // Normalize local_y by dividing by alpha
        for (int i = 0; i < local_n; i++) {
            local_y[i] /= alpha;
        }
        
        if (rank == 0) {
            //rank 0 result
            memcpy(global_y, local_y, local_n * sizeof(double));
            int offset = local_n;
            for (int p = 1; p < size; p++) {
                int count = local_n_init;
                MPI_Status status;
                MPI_Recv(global_y + offset, count, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, &status);
                offset += count;
            }
        } else {
            //other ranks send their results
            MPI_Send(local_y, local_n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
        // Broadcast global_y to all processes
        MPI_Bcast(global_y, global_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        // Check convergence: if |alpha - alpha_prev| < tol then stop
        if (k > 0 && fabs(alpha - alpha_prev) < tol) {
            break;
        }
        alpha_prev = alpha;
        
        // Update v = global_y for next iteration
        memcpy(v, global_y, global_n * sizeof(double));
    }
    
    *iter_count = k + 1;
    free(local_y);
    free(global_y);
    return alpha;
}


//D'apres le cours : Toute valeur propre de A appartient à l’un au moins des disques de Gerschgorin.
void check_gershgorin(const int n, const int *row_ptr, const int *col_idx, const double *vals,
                      const double eigenvalue)
{
    int in_a_disc = 0;
    
    for (int i = 0; i < n; i++) {
        double diag = 0.0;
        double radius = 0.0;
        for (int j = row_ptr[i]; j < row_ptr[i+1]; j++) {
            int col = col_idx[j];
            double value = vals[j];
            if (col == i) {
                diag = value; //a(i,i)
            } else {
                radius += fabs(value); //pour i fixe, sum_j a(i,j)
            }
        }
        if (fabs(eigenvalue - diag) <= radius) {
            in_a_disc++;
        }
    }
    if (in_a_disc) {
        printf("Computed eigenvalue %f lies in %d/%d Gershgorin discs.\n", eigenvalue, in_a_disc,n);
    } else {
        printf("Computed eigenvalue %f is not in any Gershgorin disc!\n", eigenvalue);
    }
}




int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        printf("Total number of processes: %d\n", size);
    }
    
    // Process 0 reads the global CSR matrix and constructs the global vector
    int *global_row_ptr = NULL, *global_col_idx = NULL;
    double *global_vals = NULL;
    int global_n, global_num_values;
    double *v = NULL;  // global vector
    
    if (rank == 0) {
        char *filename = "bcsstk03.mtx";
        constructCSR(filename, &global_row_ptr, &global_col_idx, &global_vals, &global_n, &global_num_values);
        v = constructVector(global_n);
    }
    
    // Broadcast global_n to all processes
    MPI_Bcast(&global_n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    
    // Distribute the CSR matrix
    int local_n_init = global_n / size;
    int remainder = global_n % size;
    int local_n, local_start;
    if (rank == 0) {
        local_n = local_n_init + remainder;
        local_start = 0;
    } else {
        local_n = local_n_init;
        local_start = local_n_init + remainder + (rank - 1) * local_n_init;
    }
    
    int local_nb_values;
    int *local_row_ptr = NULL;
    int *local_col_idx = NULL;
    double *local_vals = NULL;
    
    if (rank == 0) {
        int start_idx = global_row_ptr[local_start];
        int end_idx = global_row_ptr[local_start + local_n];
        local_nb_values = end_idx - start_idx;
        local_row_ptr = (int *)malloc((local_n + 1) * sizeof(int));
        local_col_idx = (int *)malloc(local_nb_values * sizeof(int));
        local_vals = (double *)malloc(local_nb_values * sizeof(double));
        for (int i = 0; i < local_n + 1; i++) {
            local_row_ptr[i] = global_row_ptr[local_start + i] - start_idx;
        }
        for (int i = 0; i < local_nb_values; i++) {
            local_col_idx[i] = global_col_idx[start_idx + i];
            local_vals[i] = global_vals[start_idx + i];
        }
        // Process 0 sends submatrix data to other processes
        for (int p = 1; p < size; p++) {
            int p_local_n = local_n_init;
            int p_local_start = local_n_init + remainder + (p - 1) * local_n_init;
            int p_start_idx = global_row_ptr[p_local_start];
            int p_end_idx = global_row_ptr[p_local_start + p_local_n];
            int p_local_nb_values = p_end_idx - p_start_idx;
            MPI_Send(&p_local_n, 1, MPI_INT, p, 0, MPI_COMM_WORLD);
            MPI_Send(&p_local_nb_values, 1, MPI_INT, p, 0, MPI_COMM_WORLD);
            int *temp_ptr = (int *)malloc((p_local_n + 1) * sizeof(int));
            for (int i = 0; i < p_local_n + 1; i++) {
                temp_ptr[i] = global_row_ptr[p_local_start + i] - p_start_idx;
            }
            MPI_Send(temp_ptr, p_local_n + 1, MPI_INT, p, 0, MPI_COMM_WORLD);
            free(temp_ptr);
            MPI_Send(&global_col_idx[p_start_idx], p_local_nb_values, MPI_INT, p, 0, MPI_COMM_WORLD);
            MPI_Send(&global_vals[p_start_idx], p_local_nb_values, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(&local_n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&local_nb_values, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_row_ptr = (int *)malloc((local_n + 1) * sizeof(int));
        local_col_idx = (int *)malloc(local_nb_values * sizeof(int));
        local_vals = (double *)malloc(local_nb_values * sizeof(double));
        MPI_Recv(local_row_ptr, local_n + 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(local_col_idx, local_nb_values, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(local_vals, local_nb_values, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    if (rank != 0) {
        v = (double *)malloc(global_n * sizeof(double));
    }
    MPI_Bcast(v, global_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Perform distributed power iteration
    double tol = 0.0001;
    int max_iter = 1000;
    int iter_count = 0;

    double largest_eigenvalue = distributed_power_iteration(v, local_n, global_n, local_row_ptr, local_col_idx, local_vals, tol, max_iter, &iter_count);
    if (rank == 0) {
        printf("Matrix size: %d, Number of processes: %d, Iterations: %d\n", global_n, size, iter_count);
        printf("Estimated largest eigenvalue: %f\n", largest_eigenvalue);
        check_gershgorin(global_n, global_row_ptr, global_col_idx, global_vals, largest_eigenvalue);
    }
    
    if (rank == 0) {
        free(global_row_ptr);
        free(global_col_idx);
        free(global_vals);
    }
    free(local_row_ptr);
    free(local_col_idx);
    free(local_vals);
    free(v);
    
    MPI_Finalize();
    return 0;
}
