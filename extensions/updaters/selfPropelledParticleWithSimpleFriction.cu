#include "std_include.h"
#include "curand_kernel.h"
#include "indexer.h"
#include "selfPropelledParticleWithSimpleFriction.cuh"
#include "selfPropelledParticleWithSimpleFriction.h"

// __global__ void double2ToDouble_kernel(
//     const double2 *input,
//     double *output,
//     int N)
// {
//     unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= N) return;
//     output[2 * idx] = input[idx].x;
//     output[2 * idx + 1] = input[idx].y;
// }
__global__ void doubleToDouble2_kernel(
    const double *input,
    double2 *output,
    int N)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 2*N) return;
    int cellIdx = idx / 2; // Determine the cell index
    int direction = idx % 2; // Determine the direction (0 for x,
    if (direction == 1) {
        // y direction
        output[cellIdx].y = input[idx];
    } else {
        // x direction
        output[cellIdx].x = input[idx];
    }
}

__global__ void calculateForces_kernel(
    const double2 *forces,
    const double2 *motility,
    const double *cellDirectors,
    double *totalf_flat,
    int N)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 2*N) return;
    int cellIdx = idx / 2; // Determine the cell index
    int direction = idx % 2; // Determine the direction (0 for x, 1 for y)
    // Calculate the velocity vector based on motility and cell director
    double v0 = motility[cellIdx].x;
    if (direction == 1) {
        totalf_flat[idx] = v0 * sin(cellDirectors[cellIdx]) + forces[cellIdx].y;
    } else {
        totalf_flat[idx] = v0 * cos(cellDirectors[cellIdx]) + forces[cellIdx].x;
    }
}

__global__ void fillRowSize_kernel(int *row_sizes, const int *d_nn, int N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2*N) return;
    int cellidx = i / 2; // Determine the cell index
    row_sizes[i] = d_nn[cellidx] + 1; // x direction
}


__global__ void buildFrictionMatrixCSR_kernel(
    const int* __restrict__ neighborNum,
    const int* __restrict__ neighbors,
    Index2D n_idx,
    int N,
    double xi_sub,
    double xi_rel,
    int* __restrict__ row_ptr,
    int* __restrict__ col_idx,
    double* __restrict__ values)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2*N) return;
    int cellidx = i / 2; // Determine the cell index
    int direction = i % 2; // Determine the direction (0 for x, 1 for y)
    int row_start = row_ptr[i];
    int nn = neighborNum[cellidx];
    int offset = 0;
    // Off-diagonal entries (neighbors)
    for (int k = 0; k < nn; ++k) {
        int j = neighbors[n_idx(k, cellidx)];
        col_idx[row_start + offset] = (2*j) + direction; // Adjust for direction
        values[row_start + offset] = -xi_rel;
        ++offset;
    }
    // Diagonal entry
    col_idx[row_start + offset] = i;
    values[row_start + offset] = xi_sub + xi_rel * nn;
}

__global__ void spp_friction_eom_integration_kernel(
                                           double2 *displacements,
                                           const double2 *motility,
                                           double *cellDirectors,
                                           const double2 *velocities,
                                           curandState *RNGs,
                                           int N,
                                           double deltaT,
                                           int Timestep)
    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >=N)
        return;
    //update displacements
    displacements[idx].x = deltaT*velocities[idx].x;
    displacements[idx].y = deltaT*velocities[idx].y;

    //next, get an appropriate random angle displacement
    curandState_t randState;
    randState=RNGs[idx];
    double Dr = motility[idx].y;
    double angleDiff = cur_norm(&randState)*sqrt(2.0*deltaT*Dr);
    RNGs[idx] = randState;
    //update director 
    double tempTheta = cellDirectors[idx];
    // ensure that the angle is between -pi and pi
    tempTheta += angleDiff;
    tempTheta = fmod(tempTheta + M_PI, 2 * M_PI) - M_PI; // Normalize to [-pi, pi]
    // update the cell director
    cellDirectors[idx] = tempTheta;
    return;
    };

int gpu_computeRowPtr(
    const int* d_nn,
    int N,
    int* d_row_ptr,
    int* d_row_sizes)
{
    // Calculate the row sizes
    int nrows = 2 * N; // Two directions (x and y)
    int blockSize = 128;
    int nBlocks = (nrows + blockSize - 1) / blockSize;
    fillRowSize_kernel<<<nBlocks, blockSize>>>(d_row_sizes, d_nn, N);
    // Do inclusive scan using Thrust to get row_ptr
    thrust::inclusive_scan(thrust::device, d_row_sizes, d_row_sizes + nrows, d_row_ptr+1);
    int nnz = thrust::reduce(thrust::device, d_row_sizes, d_row_sizes + nrows, 0, thrust::plus<int>());
    return nnz; // Return the number of non-zero entries
}

bool gpu_spp_friction_eom_integration(   
                    const int *neighborNum,
                    const int *neighbors,
                    const double2 *forces,
                    double2 *velocities,
                    double *velocity_flat,
                    double *totalf_flat,
                    double2 *displacements,
                    const double2 *motility,
                    double *cellDirectors,
                    int *d_row_ptr,
                    int *d_col_idx,
                    double *d_values,
                    int *d_row_sizes,
                    curandState *RNGs,
                    int N,
                    Index2D n_idx,
                    double deltaT,
                    int Timestep,
                    double xi_sub,
                    double xi_rel,
                    cusolverSpHandle_t handle,
                    cusparseMatDescr_t descrA)
{
    // int* d_row_ptr;
    // int* d_col_idx;
    // double* d_values;
    // int nrows = 2 * N; // Two directions (x and y)
    // cudaMalloc(&d_row_ptr, (nrows + 1) * sizeof(int));
    int nnz = gpu_computeRowPtr(neighborNum, N, d_row_ptr,d_row_sizes);
    // cudaMalloc(&d_col_idx, nnz * sizeof(int));
    // cudaMalloc(&d_values, nnz * sizeof(double));
    int blockSize = 128;
    int nBlocks = (2*N + blockSize - 1) / blockSize;
    // Build the friction matrix in CSR format
    buildFrictionMatrixCSR_kernel<<<nBlocks, blockSize>>>(neighborNum, 
        neighbors, 
        n_idx, 
        N, 
        xi_sub, 
        xi_rel, 
        d_row_ptr, 
        d_col_idx, 
        d_values);
    // calculate the total forces vector
    calculateForces_kernel<<<nBlocks, blockSize>>>(
        forces,
        motility,
        cellDirectors,
        totalf_flat,
        N
    );
    // solve the linear system Ax=b using the cusolver
    // CuSolverHandle handle;
    // cusolverSpCreate(&handle);
    // cusparseMatDescr_t descrA;
    // cusparseCreateMatDescr(&descrA);
    // cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
    // cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    // cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT);
    // cusparseSetMatFillMode(descrA, CUSPARSE_FILL_MODE_LOWER);
    int singularity = 0;
    cusolverStatus_t solve_status = cusolverSpDcsrlsvchol(
        handle, 
        2*N,
        nnz, 
        descrA,
        d_values,
        d_row_ptr,
        d_col_idx,
        totalf_flat, // b vector
        1e-12, // tolerance
        0, // reorder
        velocity_flat, // output velocity_flat
        &singularity);
    if (solve_status != CUSOLVER_STATUS_SUCCESS) {
    std::cerr << "ERROR: cusolverSpDcsrlsvchol failed with status: " << solve_status << std::endl;
    // Depending on the error, you might want to consider the handle invalid here.
    // For example, if solve_status indicates an internal error,
    // you might set a flag to skip cusolverSpDestroy.
    // Or re-create the handle if this function is called repeatedly.
    } else {
    // std::cout << "cusolverSpDcsrlsvchol returned CUSOLVER_STATUS_SUCCESS." << handle << std::endl;
    }
    // Convert the flat displacements back to double2 format
    doubleToDouble2_kernel<<<nBlocks, blockSize>>>(
        velocity_flat,
        velocities,
        N
    );

    // integration and update
    nBlocks = (2*N + blockSize - 1) / blockSize;
    spp_friction_eom_integration_kernel<<<nBlocks, blockSize>>>(
        displacements,
        motility,
        cellDirectors,
        velocities,
        RNGs,
        N,
        deltaT,
        Timestep
    );
    return true;
}
