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

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        printf("Total number of processus: %d\n", size);
    }
    
    int *global_row_ptr = NULL, *global_col_idx = NULL;
    double *global_vals = NULL;
    int global_n, global_num_values;
    double *x = NULL;  // global vector
    
    double t_start, t_end;
    
    // Process 0 reads the matrix and constructs the global CSR matrix
    if (rank == 0) {
        char *filename = "bcsstk03.mtx";
        constructCSR(filename, &global_row_ptr, &global_col_idx, &global_vals,
                     &global_n, &global_num_values);
        x = constructVector(global_n);
    }
    
    // Broadcast matrix size to all processes
    MPI_Bcast(&global_n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Compute block partition for rows
    int local_n_init = global_n / size;
    int remainder = global_n % size;
    
    int local_n;
    int local_start; // global row index of first row for this process
    if (rank == 0) {
        local_n = local_n_init + remainder;
        local_start = 0;
    } else {
        local_n = local_n_init;
        local_start = (local_n_init + remainder) + (rank - 1) * local_n_init;
        // size of rank 0 + size of previous ranks
    }
    
    /** Distribute CSR **/
    
    int local_nb_values;
    int *local_row_ptr = NULL;
    int *local_col_idx = NULL;
    double *local_vals = NULL;
    
    if (rank == 0) {
        // Process 0 CSR
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
        
        // P0 send local CSR to other process
        for (int p = 1; p < size; p++) {
            
            int p_local_n = local_n_init;
            int p_local_start = local_n_init + remainder + (p - 1) * local_n_init;
            
            int p_start_idx = global_row_ptr[p_local_start];
            int p_end_idx = global_row_ptr[p_local_start + p_local_n];
            
            int p_local_nb_values = p_end_idx - p_start_idx;
            
            // Send local csr_n and csr_nb_values
            MPI_Send(&p_local_n, 1, MPI_INT, p, 0, MPI_COMM_WORLD);
            MPI_Send(&p_local_nb_values, 1, MPI_INT, p, 0, MPI_COMM_WORLD);
            
            // Send CSR
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
        // For rank!=0, receive CSR
        MPI_Recv(&local_n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&local_nb_values, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_row_ptr = (int *)malloc((local_n + 1) * sizeof(int));
        local_col_idx = (int *)malloc(local_nb_values * sizeof(int));
        local_vals = (double *)malloc(local_nb_values * sizeof(double));
        MPI_Recv(local_row_ptr, local_n + 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(local_col_idx, local_nb_values, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(local_vals, local_nb_values, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    
    // Broadcast global vector x to all processus
    if (rank != 0) {
        x = (double *)malloc(global_n * sizeof(double));
    }
    MPI_Bcast(x, global_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // create y_local
    double *local_y = (double *)malloc(local_n * sizeof(double));
    
    /** Compute y_local = A_local * x **/
    
    double t_local_start = MPI_Wtime();
    csr_matvec(local_n, local_row_ptr, local_col_idx, local_vals, x, local_y);
    double t_local_end = MPI_Wtime();
    double local_time = t_local_end - t_local_start;
    
    /** Assemble  y_local  **/
    double *y = NULL;
    if (rank == 0) {
        y = (double *)malloc(global_n * sizeof(double));
        memcpy(y, local_y, local_n * sizeof(double));
        int offset = local_n;
        int recv_size;
        for (int p = 1; p < size; p++) {
            MPI_Recv(&recv_size, 1, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(y + offset, recv_size, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            offset += recv_size;
        }
    } else {
        MPI_Send(&local_n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(local_y, local_n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    // DONE
    double sum_local_time;
    MPI_Reduce(&local_time, &sum_local_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Result of y = A*x is : \n");
        for (int i = 0; i < global_n; i++) {
            printf("y[%d] = %f\n", i, y[i]);
        }
        printf("Average matvec time: %f sec\n", sum_local_time/size);
        
        free(global_row_ptr);
        free(global_col_idx);
        free(global_vals);
        free(y);
    }
    
    free(local_row_ptr);
    free(local_col_idx);
    free(local_vals);
    free(local_y);
    free(x);
    
    MPI_Finalize();
    return 0;
}
