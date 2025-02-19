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
        temp[row] = temp[row]+1 ; //temp[row] = Cumulative number of values in the row
        col_idx[temp[row]] = coo_cols[i];
        vals[temp[row]] = coo_vals[i];
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

int main()
{
    int *csr_row_ptr, *csr_col_idx;
    double *csr_vals;
    int csr_n, csr_num_values;
    char *filename = "bcsstk03.mtx";
    
    // Construct Matrix
    constructCSR(filename, &csr_row_ptr, &csr_col_idx, &csr_vals, &csr_n, &csr_num_values);
    
    // Construct vector
    double *x = constructVector(csr_n);
    
    // Construct result
    double *y = (double *)malloc(csr_n * sizeof(double));
    
    // Calculate y = A * x
    csr_matvec(csr_n, csr_row_ptr, csr_col_idx, csr_vals, x, y);
    
    printf("The result of y= Ax is : \n");
    for (int i = 0; i < csr_n; i++) {
        printf("y[%d] = %f\n", i, y[i]);
    }
    
    // 释放内存
    free(csr_row_ptr);
    free(csr_col_idx);
    free(csr_vals);
    free(x);
    free(y);
    
    return 0;
}
