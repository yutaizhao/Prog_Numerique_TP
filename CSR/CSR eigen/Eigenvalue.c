#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

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
    int *vk = (int *)malloc(M * sizeof(int));
    for (int i = 0; i < M; i++) {
        vk[i] = row_ptr[i];
    }
    
    // Transfer COO data into CSR
    for (int i = 0; i < total_size; i++) {
        int row = coo_rows[i];
        int pos = vk[row]++; //vk[row] = Cumulative number of values in the row, +1 : next value
        col_idx[pos] = coo_cols[i];
        vals[pos] = coo_vals[i];
    }
    
    free(vk);
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
        vec[i] = 0.1*i;
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

/*   Donné dans le cours :
 *   Algorithm 4: Un algorithme de la méthode des puissances
 *   Input : A, v, n. Output : alpha_1
 *   1.Initialisation v1 = v/norm.inf(v)
 *   2. For k = 1,2,... jusqu’à la convergence do：
 *      - v_k = A * v_{k-1}
 *      - alpha_k = norm.inf(v_k)
 *      - v_k = v_k/alpha_k
 *      - if |alpha_k - alpha_{k-1}| < tol stop
 *   3. alpha_1 = alpha_k
 */

double power_iteration(double *v, const int *row_ptr, const int *col_idx, const double *vals,
                       const int n, const double tol, const int max_iter)
{
    double alpha_prev = 0.0;
    double alpha = 0.0;
    double *vk = (double *)malloc(n * sizeof(double));
    
    //1.Initialisation v1 = v/norm.inf(v)
    double norm_inf = 0.0;
    for (int i = 0; i < n; i++) {
        double abs_val = fabs(v[i]);
        if (abs_val > norm_inf)
            norm_inf = abs_val;
    }
    for (int i = 0; i < n; i++) {
        v[i] /= norm_inf;
    }
    
    for (int k = 0; k < max_iter; k++) {
        
        //2.v_k = A * v_{k-1}
        csr_matvec(n, row_ptr, col_idx, vals, v, vk);
        
        //2.alpha_k = norm.inf(v_k)
        norm_inf = 0.0;
        for (int i = 0; i < n; i++) {
            double abs_val = fabs(vk[i]);
            if (abs_val > norm_inf)
                norm_inf = abs_val;
        }
        alpha = norm_inf;
        
        //2.v_k = vk / alpha
        for (int i = 0; i < n; i++) {
            vk[i] /= alpha;
        }
        
        //2.if |alpha_k - alpha_{k-1}| < tol stop
        if (k > 0 && fabs(alpha - alpha_prev) < tol) {
            break;
        }
        alpha_prev = alpha;
        
        //Vecteur propre
        for (int i = 0; i < n; i++) {
            v[i] = vk[i];
        }
    }
    
    free(vk);
    //3.alpha_1 = alpha_k
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


int main()
{
    int *csr_row_ptr, *csr_col_idx;
    double *csr_vals;
    int csr_n, csr_num_values;
    char *filename = "bcsstk03.mtx";
    
    constructCSR(filename, &csr_row_ptr, &csr_col_idx, &csr_vals, &csr_n, &csr_num_values);
    
    double *v = constructVector(csr_n);
    
    double tol = 0.0001;
    int max_iter = 1000;
    
    double largest_eigenvalue = power_iteration(v, csr_row_ptr, csr_col_idx, csr_vals, csr_n, tol, max_iter);
    printf("Estimated largest eigenvalue: %f\n", largest_eigenvalue);
    
    check_gershgorin(csr_n, csr_row_ptr, csr_col_idx, csr_vals, largest_eigenvalue);
    
    free(csr_row_ptr);
    free(csr_col_idx);
    free(csr_vals);
    free(v);
    
    return 0;
}
