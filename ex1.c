#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void build_csr_stencil(int N, int **row_ptr, int **col_idx, double **values) {
    
    int size = 4*3 + 4*4*(N-2) + 5 * (N-2)^2  ;
    /*
     4 * 1 corner elements * (1 center + 2 neibors) ,
     4 * (N-2) edge points * (1 center + 3 neibors ),
     (N-2)*(N-2) interior elements * (1 center + 4 neibors)
     */
    
    *row_ptr = (int*)    malloc((N*N + 1) * sizeof(int)); //for row i, values[row_ptr[i]:row_ptr[i+1]-1] = the values in the row
    *col_idx = (int*)    malloc(size * sizeof(int)); //colmun of values
    *values  = (double*) malloc(size * sizeof(double)); //non-zero values
    
    int ieme_value = 0;// n-ieme non-zero element
    
        for(int r = 0; r < N; r++) { //row
            for(int c = 0; c < N; c++) { //column
                int i = r*N + c; //the element i 2D index in 1D

                
                (*row_ptr)[i] = ieme_value;
                (*col_idx)[ieme_value] = i; //value is situated at (row_num ,col_idx[values_idx])
                (*values)[ieme_value]  = 4.0;
                ieme_value++; //a value has been added, so update to i+1-eme value

                
                /*
                 Then we evaluate the case where the element i has a -1 element on the top/down/left/right
                 */
                
                // we are at row 2~N, so the element i has upper row
                if(r > 0) {
                    int up_i = (r - 1)*N + c;
                    (*col_idx)[ieme_value] = up_i;
                    (*values)[ieme_value]  = -1.0;
                    ieme_value++;
                }

                // we are at row 1~N-1, so the element i has lower row
                if(r < N - 1) {
                    int down_i = (r + 1)*N + c;
                    (*col_idx)[ieme_value] = down_i;
                    (*values)[ieme_value]  = -1.0;
                    ieme_value++;
                }

                // we are at column 2~N, so the element i has left column
                if(c > 0) {
                    int left_i = r*N + (c - 1);
                    (*col_idx)[ieme_value] = left_i;
                    (*values)[ieme_value]  = -1.0;
                    ieme_value++;
                }

                // we are at column 1~N-1, so the element i has right column
                if(c < N- 1) {
                    int right_i = r*N + (c + 1);
                    (*col_idx)[ieme_value] = right_i;
                    (*values)[ieme_value]  = -1.0;
                    ieme_value++;
                }
            }
            (*row_ptr)[N*N] = ieme_value;
        }

    //cedric.chevalier@cea.fr
    
    
    
}


int main() {
    int N = 4;  // For example, a 4x4 grid corresponds to a 16x16 matrix

    int *row_ptr = NULL;
    int *col_idx = NULL;
    double *vals = NULL;

    build_csr_stencil(N, &row_ptr, &col_idx, &vals);

    int dim = N * N;  // The matrix size is (N^2) x (N^2)

    // Allocate a dense matrix (dim x dim) for visualization
    double **dense = (double**) malloc(dim * sizeof(double*));
    for(int i = 0; i < dim; i++) {
        dense[i] = (double*) calloc(dim, sizeof(double));
        // calloc ensures that all entries are initialized to zero
    }

    // Copy the CSR content into the dense matrix
    // For each row i, its nonzero columns are in the range [ row_ptr[i], row_ptr[i+1] )
    for(int i = 0; i < dim; i++) {
        int start = row_ptr[i];
        int end   = row_ptr[i+1];
        for(int k = start; k < end; k++) {
            int j = col_idx[k];      // row i, column j
            double val = vals[k];
            dense[i][j] = val;
        }
    }

    printf("The %dx%d matrix in dense form:\n", dim, dim);
    for(int i = 0; i < dim; i++) {
        for(int j = 0; j < dim; j++) {
            printf("%6.2f ", dense[i][j]);
        }
        printf("\n");
    }

    for(int i = 0; i < dim; i++) {
        free(dense[i]);
    }
    free(dense);

    free(row_ptr);
    free(col_idx);
    free(vals);

    return 0;
}
