#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#include "HYPRE_utilities.h"
#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"

double max(double arr[], int size) {
    double max_val = arr[0];
    for (int i = 1; i < size; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
        }
    }
    return max_val;
}

/** Les parametres **/
typedef struct {
    double k0;         // k(u) = k0 * u^q
    double sigma;      // radiation constant
    double beta;       // for Q(x)
    double q;          // for k(u)
    double delta;      // l'épaisseur δ de la flamme
    double L;          // Une distance grande par rapport δ
    int    method;     // 1 => linearise,  2 => Newton
} ProblemParams;


/** k(u) **/
double k(double u, const ProblemParams *params)
{
    return params->k0 * pow(u, params->q);
}

/** Q(x) **/
double Q(double x, const ProblemParams *params) {
    double sourceTerm = 0.0;
    if (x <= params->delta) {
        sourceTerm = params->beta;
    }
    return sourceTerm;
}

/*
 * LinearizedImplicit A with A*ui^{n+1} = ui ^n + dt(Qi + sigma)
 * NB : u'(0)=0 and u(N)=1
 */
void BuildMatrixAndRhs_LinearizedImplicit( HYPRE_IJMatrix A, HYPRE_IJVector b, const double *u_old,
                                          int N, double dx, double dt, const ProblemParams *params )
{
    double *rhs_local = (double*) calloc(N+1, sizeof(double));
    
    int    col[3];
    double val[3];
    
    for(int i = 0; i <= N; i++)
    {
        int row_index = i;
        
        // the Dirichlet condition : A(N,N)=1, b(N) = u(x_N)=1
        if(i == N)
        {
            int ncols = 1;
            col[0] = i;
            val[0] = 1.0;
            HYPRE_IJMatrixSetValues(A, 1, &ncols, &row_index, col, val);
            
            rhs_local[i] = 1.0;
        }
        // the Neumann condition at i = 0,
        // translated as a mirror condition for the equation :  u'(0)=0 => u_0=u(x_0)=u(x_1)
        else if(i == 0)
        {
            double u0 = u_old[0];
            double u1 = u_old[1];
            
            double a0 = 1.0 + dt*( k(0.5*(u0 + u1), params)/(dx*dx) + 4.0*params->sigma*pow(u0,3.0) );
            double b0 = -dt*(k(0.5*(u0 + u1), params)/(dx*dx));
            
            int ncols = 2;
            col[0] = 0;
            val[0] = a0;
            col[1] = 1;
            val[1] = b0;
            HYPRE_IJMatrixSetValues(A, 1, &ncols, &row_index, col, val);
            
            // RHS
            rhs_local[i] = u0 + dt*(Q(i*dx, params) + params->sigma);
        }
        else
        {
            // i except boundry
            double ui   = u_old[i];
            double uim1 = u_old[i-1];
            double uip1 = u_old[i+1];
            
            double kim_demi  = k(0.5*(ui + uim1), params); //k_{i-1/2}
            double kip_demi = k(0.5*(ui + uip1), params); //k_{i+1/2}
            
            double ai = 1.0 + dt*( (kim_demi + kip_demi)/(dx*dx) + 4.0*params->sigma*pow(ui,3.0) );
            double bi = -dt*( kip_demi/(dx*dx) );
            double ci = -dt*( kim_demi/(dx*dx) );
            
            int ncols = 3;
            col[0] = i-1;
            val[0] = ci;
            col[1] = i;
            val[1] = ai;
            col[2] = i+1;
            val[2] = bi;
            
            HYPRE_IJMatrixSetValues(A, 1, &ncols, &row_index, col, val);
            
            // RHS
            rhs_local[i] = ui + dt*(Q(i*dx, params) + params->sigma);
        }
    }
    
    // write rhs_local to b
    {
        int nrows = N+1;
        int *rows = (int*) calloc(nrows, sizeof(int));
        for(int i=0; i<=N; i++){
            rows[i] = i;
        }
        HYPRE_IJVectorSetValues(b, nrows, rows, rhs_local);
        free(rows);
    }
    
    free(rhs_local);
}


/*
 * Newton J(u) with J(u)*(ui^{k+1}-ui^{k}) = -F(u)
 */
void BuildMatrixAndRhs_Newton( HYPRE_IJMatrix A, HYPRE_IJVector b, const double *u_current,
                              int N, double dx, const ProblemParams *params )
{
    double *rhs_F = (double*) calloc(N+1, sizeof(double));
    int i, col[3];
    double val[3];
    
    for(i = 0; i <= N; i++)
    {
        int row_index = i;
        
        //For the boudries (0 and N), we know that :
        //For x=0=1=N, u are constant, indeed :
        //u(x_0) =u(x_1) and u(x_N) =1
        //If we insert the values into the formula, we can remark that F(u)=0
        if(i == N)
        {
            int ncols = 1;
            col[0] = i;
            val[0] = 1.0;
            HYPRE_IJMatrixSetValues(A, 1, &ncols, &row_index, col, val);
            rhs_F[i] = 0;
        }
        else if(i == 0)
        {
            
            int ncols = 2;
            col[0] = 0;
            val[0] =  1.0;
            col[1] = 1;
            val[1] = -1.0;
            HYPRE_IJMatrixSetValues(A, 1, &ncols, &row_index, col, val);
            rhs_F[i] = 0;
        }
        else
        {
            // for i = [1..N-1[
            double ui   = u_current[i];
            double uim1 = u_current[i-1];
            double uip1 = u_current[i+1];
            
            double kim_demi = k(0.5*(ui + uim1), params);
            double kip_demi = k(0.5*(ui + uip1), params);
            
            /** Compute F_i(u) : **/
            
            // diffTerm = [kip_demi*(uip1 - ui) - kim_demi*(ui - uim1)]/(dx*dx)
            double diffTerm = (kip_demi*(uip1 - ui)- kim_demi*(ui - uim1))/(dx*dx) ;
            
            // radTerm = sigma * (u_i^4 - 1)
            double radTerm = params->sigma * ( pow(ui,4.0) - 1.0 );
            
            // F_i = - diffTerm + radTerm - Q(x_i)
            double x_i = i * dx;
            double Fi = -diffTerm + radTerm - Q(x_i, params);
            
            rhs_F[i] = Fi;
            
            /** Compute Jacobian matrix **/
            
            double ci = - kim_demi/(dx*dx) ;
            double bi = - kip_demi/(dx*dx) ;
            double ai = -ci - bi + params->sigma*4.0*pow(ui,3.0);
            
            int ncols = 3;
            col[0] = i-1;
            val[0] = ci;
            col[1] = i;
            val[1] = ai;
            col[2] = i+1;
            val[2] = bi;
            HYPRE_IJMatrixSetValues(A, 1, &ncols, &row_index, col, val);
            
        }
    }
    
    //b = -F(u)
    {
        int *rows = (int*) calloc(N+1, sizeof(int));
        for(int r=0; r<=N; r++){
            rows[r] = r;
            rhs_F[r] = -rhs_F[r];
        }
        
        HYPRE_IJVectorSetValues(b, N+1, rows, rhs_F);
        free(rows);
    }
    
    free(rhs_F);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int myrank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    // Set problem and discretization parameters
    ProblemParams params;
    params.k0    = 0.01;
    params.sigma = 0.1;
    params.beta  = 1.0;
    params.q     = 0.5;
    params.delta = 0.2;
    params.L     = 1.0;
    
    // Choose method: 1 => Linearized implicit, 2 => Newton
    params.method = 1;
    
    // Define the list of discretization sizes and gamma values
    int N_list[] = {50, 100, 500, 1000, 5000, 10000};
    int numN = sizeof(N_list) / sizeof(N_list[0]);
    double gamma_list[] = {0.1, 1.0, 10.0};
    int numGamma = sizeof(gamma_list) / sizeof(gamma_list[0]);
    
    // Loop over each discretization
    for (int iN = 0; iN < numN; iN++)
    {
        int N = N_list[iN];
        double dx = params.L / (double) N;
        
        double *u = (double*) malloc((N+1) * sizeof(double));
        for (int i = 0; i <= N; i++) {
            u[i] = 1.0;
        }
        
        // Loop over gamma values
        for (int ig = 0; ig < numGamma; ig++)
        {
            double gamma = gamma_list[ig];
            // Assume initial u_max = 1.0 (since u is initialized to 1)
            double u_max = 1.0;
            double dt = gamma * 2.0 / (4.0 * params.sigma * pow(u_max, 3.0) + 4.0 * k(u_max, &params) / (dx*dx));
            
            int maxIters = 1000;
            
            // Create hypre IJMatrix and IJVectors (A, b, x)
            HYPRE_IJMatrix A;
            HYPRE_IJVector b, x;
            int ilower = 0, iupper = N;
            
            HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &A);
            HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
            HYPRE_IJMatrixInitialize(A);
            
            HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &b);
            HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
            HYPRE_IJVectorInitialize(b);
            
            HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &x);
            HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
            HYPRE_IJVectorInitialize(x);
            
            // Main iterative solver loop
            for (int iter = 0; iter < maxIters; iter++)
            {
                // Clear A, b, x for the current iteration
                HYPRE_IJMatrixSetConstantValues(A, 0.0);
                HYPRE_IJVectorSetConstantValues(b, 0.0);
                HYPRE_IJVectorSetConstantValues(x, 0.0);
                
                // Assemble matrix and RHS according to the chosen method
                if (params.method == 1)
                    BuildMatrixAndRhs_LinearizedImplicit(A, b, u, N, dx, dt, &params);
                else
                    BuildMatrixAndRhs_Newton(A, b, u, N, dx, &params);
                
                HYPRE_IJMatrixAssemble(A);
                HYPRE_IJVectorAssemble(b);
                HYPRE_IJVectorAssemble(x);
                
                // Convert to ParCSR format
                HYPRE_ParCSRMatrix parA;
                HYPRE_ParVector parB, parX;
                HYPRE_IJMatrixGetObject(A, (void**) &parA);
                HYPRE_IJVectorGetObject(b, (void**) &parB);
                HYPRE_IJVectorGetObject(x, (void**) &parX);
                
                // Create the solver
                HYPRE_Solver solver, precond;
                HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver);
                HYPRE_PCGSetTol(solver, 0.00001);
                HYPRE_PCGSetMaxIter(solver, 2000);
                
                // Configure BoomerAMG as the preconditioner
                HYPRE_BoomerAMGCreate(&precond);
                HYPRE_BoomerAMGSetPrintLevel(precond, 0);
                HYPRE_BoomerAMGSetMaxIter(precond, 1);
                HYPRE_ParCSRPCGSetPrecond(solver, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup, precond);
                
                // Setup and solve the linear system
                HYPRE_ParCSRPCGSetup(solver, parA, parB, parX);
                HYPRE_ParCSRPCGSolve(solver, parA, parB, parX);
                
                // Retrieve the solution into xLocal
                double *xLocal = (double*) calloc(N+1, sizeof(double));
                {
                    int nrows = N+1;
                    int *rows = (int*) calloc(nrows, sizeof(int));
                    for (int i = 0; i <= N; i++)
                        rows[i] = i;
                    HYPRE_IJVectorGetValues(x, nrows, rows, xLocal);
                    free(rows);
                }
                
                // Update the solution vector u
                if (params.method == 1)
                {
                    for (int i = 0; i <= N; i++) {
                        u[i] = xLocal[i];
                    }
                    u_max = max(u,N+1);
                    dt = gamma * 2.0 / (4.0 * params.sigma * pow(u_max, 3.0) + 4.0 * k(u_max, &params) / (dx*dx));
                }
                else
                {
                    for (int i = 0; i <= N; i++) {
                        u[i] += xLocal[i];
                    }
                }
                free(xLocal);
                
                HYPRE_ParCSRPCGDestroy(solver);
                HYPRE_BoomerAMGDestroy(precond);
                
            } // end iteration loop
            
            HYPRE_IJMatrixDestroy(A);
            HYPRE_IJVectorDestroy(b);
            HYPRE_IJVectorDestroy(x);
            
            // Output the solution to a file
            if (myrank == 0)
            {
                char filename[256];
                sprintf(filename, "solution_N%d_gamma%.1f.dat", N, gamma);
                FILE *fout = fopen(filename, "w");
                double xcoord;
                for (int i = 0; i <= N; i++)
                {
                    xcoord = params.L * ((double) i / (double) N);
                    fprintf(fout, "%.6f  %.12f\n", xcoord, u[i]);
                }
                fclose(fout);
                printf("Final solution for N=%d, gamma=%.1f is saved to %s\n", N, gamma, filename);
            }
            
            // Reinitialize u to the initial guess for the next gamma run
            for (int i = 0; i <= N; i++)
                u[i] = 1.0;
        } // end gamma loop
        
        free(u);
    } // end N loop
    
    MPI_Finalize();
    return 0;
    
}
