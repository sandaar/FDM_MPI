# FDM_MPI
Finite Difference Method : parallel implementation using MPI

Solution for non-stationary temperature distribution in a two-dimensional rectangular plate.
The Dirichlet (or first-type) and the Neumann (or second-type) boundary conditions may be applied.
Gaussian elimination was used as an algorithm for solving systems of linear equations. Yes, there is a lot of room for optimization, I know :)

Compile the program

    mpicc mpi_fdm.c

Run the program on 8 (or 1, or 2, or 4, whatever you want) nodes:

    mpiexec -l -n 8 ./a.out

Watch how temperature changes:

    gnuplot -persist plot
