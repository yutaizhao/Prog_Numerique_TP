#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void build_csr_stencil(const int N, int **row_ptr, int **col_idx, double **values) {
    
    int size = 4*3 + 4*4*(N-2) + 5 * (N-2)*(N-2)  ;
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
        }
    
    (*row_ptr)[N*N] = ieme_value;
}

// Create an example b vector
void build_rhs(const int n, double **b){
    *b = (double*) malloc(n * sizeof(double));
    for(int i = 0; i < n; i++) {
        (*b)[i] = 0.1*i;
    }
}

void extract_diagonal(const int n, const int* row_ptr, const int* col_idx,
                      const double* vals,
                      double *diag)
{
    
    for(int i = 0; i < n; i++) {
        // Loop over the nonzeros in row i
        int start = row_ptr[i];
        int end   = row_ptr[i+1];
        for(int k = start; k < end; k++) {
            if(col_idx[k] == i) {
                diag[i] = vals[k];
                break;   // We found the diagonal element in this row
            }
        }
    }
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

// Residu = b_exacte - b_exp
void compute_residual(const int n, const int *row_ptr, const int *col_idx,
                      const double *vals, const double *x, const double *b,
                      double *r)
{
    double *b_exp = (double*) malloc(n * sizeof(double));
    csr_matvec(n, row_ptr, col_idx, vals, x, b_exp);

    // r = b - Ax
    for(int i = 0; i < n; i++) {
        r[i] = b[i] - b_exp[i];
    }
    free(b_exp);
}

double norm2(const double *v, const int n) {
    double sum = 0.0;
    for(int i = 0; i < n; i++){
        sum += v[i]*v[i];
    }
    return sqrt(sum);
}


double *jacobi_method(const int n, const int *row_ptr, const int *col_idx,
                      const double *vals, const double *b,
                      const int max_iter, const double tol)
{
    
    double *x_k = (double*) calloc(n, sizeof(double));
    double *x_k1 = (double*) calloc(n, sizeof(double));
    double *r = (double*) malloc(n * sizeof(double));
    double *diag = (double*) malloc(n * sizeof(double));
    extract_diagonal(n, row_ptr, col_idx, vals, diag);

    for(int it = 0; it < max_iter; it++) {
        
        // One Jacobi iteration:
        for(int i = 0; i < n; i++) {
            double row_sum = 0.0;
            int start = row_ptr[i];
            int end   = row_ptr[i+1];
            for(int k = start; k < end; k++) {
                int j = col_idx[k];
                if(j != i) {
                    row_sum += vals[k] * x_k[j];
                }
            }
            x_k1[i] = (b[i] - row_sum) / diag[i];
        }

        // Compute residual r = b - A*x_new
        compute_residual(n, row_ptr, col_idx, vals, x_k1, b, r);
        double b_norm = norm2(b, n);
        double r_norm = norm2(r, n);
        double res_relatif = r_norm / b_norm;
        printf("Jacobi %3d:  relative residual = %e\n", it, res_relatif);

        // Check convergence
        if(res_relatif < tol) {
            printf("Jacobi converged at iteration %d\n", it);
            free(diag);
            free(x_k);
            free(r);
            return x_k1;
        }
        
        // Not CV : update x_k <- x_k+1 and ccontinue
        for(int i=0; i<n; i++){
            x_k[i] = x_k1[i];
        }
    }
    
    printf("Jacobi did not converge and reached the max iteration\n");
    free(diag);
    free(x_k);
    free(r);
    return x_k1;
}

double *gs_method(const int n, const int *row_ptr, const int *col_idx,
                  const double *vals, const double *b,
                  const int max_iter, const double tol)
{

    double *x_k  = (double*) calloc(n, sizeof(double));
    double *x_k1 = (double*) calloc(n, sizeof(double));
    double *r = (double*) malloc(n * sizeof(double));
    double *diag = (double*) malloc(n * sizeof(double));
    extract_diagonal(n, row_ptr, col_idx, vals, diag);

    for(int it = 0; it < max_iter; it++) {

        // One Gauss–Seidel iteration
        for(int i = 0; i < n; i++) {
            double row_sum = 0.0;
            int start = row_ptr[i];
            int end   = row_ptr[i+1];
            for(int k = start; k < end; k++) {
                int j = col_idx[k];
                if(j != i) {
                    if(j < i) {
                        row_sum += vals[k] * x_k1[j];
                    } else if (j>i) {
                        row_sum += vals[k] * x_k[j];
                    }
                }
            }
            x_k1[i] = (b[i] - row_sum) / diag[i];
        }

        // Compute residual r = b - A*x_k1
        compute_residual(n, row_ptr, col_idx, vals, x_k1, b, r);

        double b_norm = norm2(b, n);
        double r_norm = norm2(r, n);
        double res_relatif = r_norm / b_norm;

        printf("Gauss-Seidel %3d:  relative residual = %e\n", it, res_relatif);

        // Check convergence
        if(res_relatif < tol) {
            printf("Gauss-Seidel converged at iteration %d\n", it);
            free(x_k);
            free(r);
            free(diag);
            return x_k1;
        }

        // Not converged: copy x_k1 -> x_k
        for(int i = 0; i < n; i++){
            x_k[i] = x_k1[i];
        }
    }

    // If we exit the loop, it means we didn't converge in max_iter
    printf("Gauss-Seidel did not converge and reached the max iteration\n");
    free(x_k);
    free(r);
    free(diag);
    return x_k1;
}

/*
 On a fait une analogie avec l'algoriithme de wikipedia,
 les numeros correspondent aux lignes de l'algo.
 */
double *cg_method(const int n, const int *row_ptr, const int *col_idx,
                  const double *vals, const double *b,
                  const int max_iter, const double tol) {
    // Initialisation
    double *x = (double *)calloc(n, sizeof(double));
    double *r = (double *)malloc(n * sizeof(double));
    double *p = (double *)malloc(n * sizeof(double)); // Direction de recherche
    double *Ap = (double *)malloc(n * sizeof(double)); // Produit matrice-vecteur A*p

    // 1 : r0 = b - A*x0
    compute_residual(n, row_ptr, col_idx, vals, x, b, r);
    
    
    // 2 : if r_0 is small
    double b_norm = norm2(b, n);
    double r_norm = norm2(r, n);
    double res_relatif = r_norm / b_norm;
    if (res_relatif < tol) {
        printf("Conjugate Gradient converged at iteration 0\n");
        free(r);
        free(p);
        free(Ap);
        return x;
    }

    // 3 : p0 = r0
    for (int i = 0; i < n; i++) {
        p[i] = r[i];
    }

    // 4 : k=0 and repeat
    for (int k = 0; k < max_iter; k++) {
        
        // 5 : rTr
        double rTr = 0.0;
        for (int i = 0; i < n; i++) {
                rTr += r[i] * r[i];
        }

        // 5 : ApA
        csr_matvec(n, row_ptr, col_idx, vals, p, Ap); // Ap = A * p
        double pAp = 0.0;
        for (int i = 0; i < n; i++) {
            pAp += p[i] * Ap[i];
        }
        
        // 5 : a_k = (r^T * r) / (p^T * A * p)
        double a_k = rTr / pAp;

        // 6 : x_k+1 = x_k + a_k * p_k
        for (int i = 0; i < n; i++) {
            x[i] += a_k * p[i];
        }

        // 7 : r_k+1 = r_k - a_k * A * p_k
        for (int i = 0; i < n; i++) {
            r[i] -= a_k * Ap[i];
        }
        
        // 8 : if r_k+1 is small
        double b_norm = norm2(b, n);
        double r_norm = norm2(r, n);
        double res_relatif = r_norm / b_norm;
        
        printf("Conjugate Gradient %3d:  relative residual = %e\n", k, res_relatif);
        
        if (res_relatif < tol) {
            printf("Conjugate Gradient converged at iteration %d\n", k);
            break;
        }
        
        // 9 : rTr_new
        double rTr_new = 0.0;
        for (int i = 0; i < n; i++) {
            rTr_new += r[i] * r[i];
        }

        // 9 : b_k = (r^T * r)_{k+1} / (r^T * r)_k
        double b_k = rTr_new / rTr;

        // 10 : p_k+1 = r_k+1 + b_k * p_k
        for (int i = 0; i < n; i++) {
            p[i] = r[i] + b_k * p[i];
        }
    }
    
    free(r);
    free(p);
    free(Ap);
    return x;
}

// Compute the dot product of two vectors
double dotprod(const double *x, const double *y, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        s += x[i] * y[i];
    }
    return s;
}

/*
 For QR decomposition, apply Givens rotations to the k-th column of matrix H (in-place),
 and update the g vector accordingly.
 H    Pointer to the (max_iter+1) x max_iter array representing the upper Hessenberg matrix.
 g    Pointer to the vector (length max_iter+1) storing the RHS of the least-squares problem.
 c    Array of cosines for Givens rotations (length max_iter).
 s    Array of sines for Givens rotations (length max_iter).
 k    The current column index (0-based) that we want to eliminate the entry below the diagonal.
 */
static void QR_givens(double *H, double *g,
                                   double *c, double *s,
                                   int k, int max_iter)
{
    // We use the same indexing as in the Arnoldi loop:
    // H[row + col*(max_iter+1)]
    
    // Apply all previous Givens rotations
    for (int i = 0; i < k; i++) {
        // row=i, col=k
        int index_ik   = i   + k*(max_iter+1);  // H[i, k]
        int index_ip1k = i+1 + k*(max_iter+1);  // H[i+1, k]

        double temp  =  c[i] * H[index_ik] + s[i] * H[index_ip1k];
        double temp2 = -s[i] * H[index_ik] + c[i] * H[index_ip1k];

        H[index_ik]   = temp;
        H[index_ip1k] = temp2;
    }

    // Compute a new Givens rotation for the entry H[k+1, k]
    // row=k,   col=k => index_kk
    // row=k+1, col=k => index_kp1k
    int index_kk   = k   + k*(max_iter+1);
    int index_kp1k = k+1 + k*(max_iter+1);

    double h_k   = H[index_kk];
    double h_k1  = H[index_kp1k];
    double denom = sqrt(h_k*h_k + h_k1*h_k1);

    if (fabs(denom) < 1e-14) {
        c[k] = 1.0;
        s[k] = 0.0;
    } else {
        c[k] = h_k   / denom;
        s[k] = h_k1  / denom;
    }

    // Make H[k+1, k] = 0
    H[index_kk]   = c[k]*h_k + s[k]*h_k1;  // effectively denom
    H[index_kp1k] = 0.0;

    // Also apply this rotation to g
    double temp  =  c[k]*g[k] + s[k]*g[k+1];
    double temp2 = -s[k]*g[k] + c[k]*g[k+1];
    g[k]   = temp;
    g[k+1] = temp2;
}


/*
 * The procedure:
 *  1. Initialize x_0 (here we use x_0 = 0).
 *  2. Compute r_0 = b - A*x_0 (here it is just b if x_0=0).
 *  3. Normalize r_0 => v_0, and set g[0] = ||r_0||.
 *  4. Arnoldi iteration to build Krylov basis V and upper Hessenberg H.
 *  5. Do QR decomposition by Givens rotations to maintain H in upper-triangular form after each new column is added.
 *  6. Check residual after each iteration. If below tol, stop.
 *  7. After finishing or reaching max_iter, solve the small upper-triangular system and update x.
 */
double* gmres_method(const int n,
                     const int *row_ptr, const int *col_idx, const double *vals,
                     const double *b,
                     const int max_iter, const double tol)
{
    // Allocate the solution vector x and initialize to zero
    double *x = (double*) calloc(n, sizeof(double));

    // Compute initial residual r0 = b - A*x0. Here x0=0 => r0 = b
    double *r = (double*) malloc(n * sizeof(double));
    for(int i = 0; i < n; i++) {
        r[i] = b[i];
    }

    // Compute beta = ||r_0||
    double beta = 0.0;
    for(int i = 0; i < n; i++){
        beta += r[i]*r[i];
    }
    beta = sqrt(beta);

    // For safety, if b is nearly zero, set the norm to 1 to avoid dividing by zero
    double b_norm = (fabs(beta) < 1e-15) ? 1.0 : beta;

    // Check initial residual
    double res_rel = beta / b_norm;
    if(res_rel < tol) {
        printf("GMRES converged at iteration 0, residual = %e\n", res_rel);
        free(r);
        return x;
    }

    // Allocate memory for the Arnoldi basis (V) and the Hessenberg matrix (H)
    // V will be (n x (max_iter+1)), flattened in row-major or column-major
    double *V = (double*) calloc((max_iter+1)*n, sizeof(double));

    // H will be ((max_iter+1) x max_iter)
    double *H = (double*) calloc((max_iter+1)*max_iter, sizeof(double));

    // Arrays for Givens rotation parameters
    double *c = (double*) calloc(max_iter, sizeof(double));  // cosines
    double *s = (double*) calloc(max_iter, sizeof(double));  // sines

    // g vector in the least squares problem (size max_iter+1)
    double *g = (double*) calloc(max_iter+1, sizeof(double));
    // g[0] = beta, others = 0
    g[0] = beta;

    // Normalize r0 => v0
    for(int i = 0; i < n; i++) {
        V[i] = r[i] / beta;  // v0
    }

    // Arnoldi Iteration
    int k;
    for(k = 0; k < max_iter; k++)
    {
        // 1) w = A * v_k
        double *w = (double*) calloc(n, sizeof(double));
        csr_matvec(n, row_ptr, col_idx, vals, &V[k*n], w);

        // 2) Modified Gram-Schmidt: for j=0..k
        for(int j = 0; j <= k; j++) {
            double dot = 0.0;
            // Compute dot = v_j^T * w
            for(int p = 0; p < n; p++){
                dot += V[j*n + p]*w[p];
            }
            // H[j, k] = dot
            H[j + k*(max_iter+1)] = dot;
            // w = w - dot * v_j
            for(int p = 0; p < n; p++){
                w[p] -= dot*V[j*n + p];
            }
        }

        // 3) h_{k+1,k} = ||w||
        double w_norm = norm2(w, n);
        H[(k+1) + k*(max_iter+1)] = w_norm;

        // If w_norm == 0, we can stop early
        if(w_norm < 1e-14) {
            free(w);
            // Update the k-th column of H with existing Givens rotations
            QR_givens(H, g, c, s, k, max_iter);

            // Check the new residual => |g[k+1]|
            double new_res = fabs(g[k+1]);
            double rel_res = new_res / b_norm;
            printf("GMRES %3d: relative residual = %e\n", k, rel_res);
            if(rel_res < tol) {
                printf("GMRES converged at iteration %d\n", k);
            }
            break;
        }

        // 4) v_{k+1} = w / w_norm
        for(int p = 0; p < n; p++){
            V[(k+1)*n + p] = w[p] / w_norm;
        }
        free(w);

        // 5) Apply Givens rotations to the k-th column => maintain upper-triangular
        QR_givens(H, g, c, s, k, max_iter);

        // 6) Check the residual => |g[k+1]|
        double new_res = fabs(g[k+1]);
        double rel_res = new_res / b_norm;
        printf("GMRES %3d: relative residual = %e\n", k, rel_res);

        if(rel_res < tol) {
            printf("GMRES converged at iteration %d\n", k);
            // Increase k so we know how many steps we did for the back-substitution
            k++;
            break;
        }
    }

    // If we finished the loop without break, might not be converged
    if(k == max_iter) {
        printf("GMRES did not converge within %d iterations.\n", max_iter);
    }

    // Now we solve the small upper-triangular system H(0..k, 0..k-1)*y = g(0..k)
    // The actual dimension is (k x k), with k <= max_iter
    int final_iter = (k > max_iter) ? max_iter : k;
    double *y = (double*) calloc(final_iter, sizeof(double));

    // Back-substitution on the triangular portion of H
    for(int i = final_iter - 1; i >= 0; i--) {
        double sum = g[i];
        for(int j = i+1; j < final_iter; j++){
            sum -= H[i + j*(max_iter+1)] * y[j];
        }
        y[i] = sum / H[i + i*(max_iter+1)];
    }

    // Construct the solution: x = x0 + V(:,0..final_iter-1)*y
    // Here x0 = 0, so x = sum_{i=0..final_iter-1} y[i] * v_i
    for(int i = 0; i < final_iter; i++){
        for(int p = 0; p < n; p++){
            x[p] += V[i*n + p] * y[i];
        }
    }

    // Free temp arrays
    free(r);
    free(V);
    free(H);
    free(g);
    free(c);
    free(s);
    free(y);

    return x;
}

int main()
{
    int N = 5;    // 4x4 grid => 16 unknowns
    int n = N*N;  // system dimension
    int max_iter = 1000;
    double tol   = 1e-6;

    // Build A (in CSR) and rhs
    int *row_ptr = NULL;
    int *col_idx = NULL;
    double *vals = NULL;
    double *b = NULL;
    build_csr_stencil(N, &row_ptr, &col_idx, &vals);
    build_rhs(n, &b);

    // Solve with Jacobi
    double *res_jac = NULL; //solution x
    res_jac = jacobi_method(n, row_ptr, col_idx, vals, b, max_iter, tol);
    // Print final solution x
    printf("Final approximate solution x using Jacobi:\n");
    for(int i = 0; i < n; i++) {
        printf("x[%d] = %g\n", i, res_jac[i]);
    }
    
    // Solve with Gauss-Seidel
    double *res_gs = NULL; //solution x
    res_gs = gs_method(n, row_ptr, col_idx, vals, b, max_iter, tol);
    // Print final solution x
    printf("Final approximate solution x using GS:\n");
    for(int i = 0; i < n; i++) {
        printf("x[%d] = %g\n", i, res_gs[i]);
    }
    
    //Solve with conjugate gradient
    double *res_cg = NULL; // solution x
    res_cg = cg_method(n, row_ptr, col_idx, vals, b, max_iter, tol);
    // Print final solution x
    printf("Final approximate solution x using CG :\n");
    for (int i = 0; i < n; i++) {
        printf("x[%d] = %g\n", i, res_cg[i]);
    }
    
    // Solve the system using GMRES
    double *res_gmres = NULL ;
    res_gmres = gmres_method(n, row_ptr, col_idx, vals, b, max_iter, tol);
    // Print the final solution
    printf("Final approximate solution using GMRES:\n");
    for (int i = 0; i < n; i++) {
        printf("x[%d] = %g\n", i, res_gmres[i]);
    }

    free(res_jac);
    free(res_gs);
    free(res_cg);
    free(res_gmres);
    free(b);
    free(row_ptr);
    free(col_idx);
    free(vals);
    return 0;
}
