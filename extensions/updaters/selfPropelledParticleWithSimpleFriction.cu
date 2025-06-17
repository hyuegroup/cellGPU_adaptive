#include "std_include.h"
#include "curand_kernel.h"
#include "indexer.h"
#include "selfPropelledParticleWithSimpleFriction.cuh"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include <math.h>
__global__ void double2ToDouble_kernel(
    const double2 *input,
    double *output,
    int N)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    output[2 * idx] = input[idx].x;
    output[2 * idx + 1] = input[idx].y;
}
__global__ void doubleToDouble2_kernel(
    const double *input,
    double2 *output,
    int N)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    output[idx].x = input[2 * idx];
    output[idx].y = input[2 * idx + 1];
}

__global__ void calculateForces_kernel(
    const double2 *forces,
    const double2 *motility,
    const double *cellDirectors,
    double2 *velocities,
    double2 *totalf,
    int N)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    // Calculate the velocity vector based on motility and cell director
    double v0 = motility[idx].x;
    velocities[idx].x = v0 * cos(cellDirectors[idx]);
    velocities[idx].y = v0 * sin(cellDirectors[idx]);
    totalf[idx].x = forces[idx].x + velocities[idx].x;
    totalf[idx].y = forces[idx].y + velocities[idx].y;
}

__global__ void fillRowSize_kernel(int *row_sizes, const int *d_nn, int N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    // Assuming each thread fills one row size
    int nn = d_nn[i];
    row_sizes[2*i] = nn + 1; // x direction
    row_sizes[2*i + 1] = nn + 1; // y direction
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
    if (i >= N) return;

    int row_start = row_ptr[i];
    int nn = neighborNum[i];
    int offset = 0;

    // Off-diagonal entries (neighbors)
    for (int k = 0; k < nn; ++k) {
        int j = neighbors[n_idx(k, i)];
        col_idx[row_start + offset] = j;
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
    displacements[idx].x = deltaT*displacements[idx].x;
    displacements[idx].y = deltaT*displacements[idx].y;

    //next, get an appropriate random angle displacement
    curandState_t randState;
    randState=RNGs[idx];
    double Dr = motility[idx].y;
    double angleDiff = cur_norm(&randState)*sqrt(2.0*deltaT*Dr);
    RNGs[idx] = randState;
    //update director 
    double currentTheta = cellDirectors[idx];
    // ensure that the angle is between -pi and pi
    if(velocities[idx].y != 0. && velocities[idx].x != 0.)
        {
        currentTheta = atan2(velocities[idx].y,velocities[idx].x);
        };
    cellDirectors[idx] = currentTheta + angleDiff;
    return;
    };

int gpu_computeRowPtr(
    const int* d_nn,
    int N,
    int* d_row_ptr)
{
    // Calculate the row sizes
    int nrows = 2 * N; // Two directions (x and y)
    int *d_row_sizes;
    cudaMalloc(&d_row_sizes, nrows * sizeof(int));
    int blockSize = 128;
    int nBlocks = (N + blockSize - 1) / blockSize;
    fillRowSize_kernel<<<nBlocks, blockSize>>>(d_row_sizes, d_nn, N);
    
    // Do exclusive scan using Thrust to get row_ptr
    thrust::device_ptr<int> t_row_sizes(d_row_sizes);
    cudaMemset(d_row_ptr, 0, sizeof(int));
    thrust::device_ptr<int> t_row_ptr(d_row_ptr);
    thrust::inclusive_scan(t_row_sizes, t_row_sizes + nrows, t_row_ptr+1);
    int nnz = thrust::reduce(t_row_sizes, t_row_sizes + nrows, 0, thrust::plus<int>());
    // Free temporary memory
    cudaFree(d_row_sizes);
    return nnz; // Return the number of non-zero entries
}

bool gpu_spp_friction_eom_integration(   
                    const int *neighborNum,
                    const double2 *forces,
                    double2 *velocities,
                    double2 *displacements,
                    const double2 *motility,
                    double *cellDirectors,
                    curandState *RNGs,
                    int N,
                    Index2D n_idx,
                    double deltaT,
                    int Timestep,
                    double xi_sub,
                    double xi_rel)
{
    int* d_row_ptr;
    int* d_col_idx;
    double* d_values;
    int nrows = 2 * N; // Two directions (x and y)
    cudaMalloc(&d_row_ptr, (nrows + 1) * sizeof(int));
    int nnz = gpu_computeRowPtr(neighborNum, N, d_row_ptr);
    cudaMalloc(&d_col_idx, nnz * sizeof(int));
    cudaMalloc(&d_values, nnz * sizeof(double));
    int blockSize = 128;
    int nBlocks = (N + blockSize - 1) / blockSize;
    // Build the friction matrix in CSR format
    buildFrictionMatrixCSR_kernel<<<nBlocks, blockSize>>>(neighborNum, 
        d_col_idx, 
        n_idx, 
        N, 
        xi_sub, 
        xi_rel, 
        d_row_ptr, 
        d_col_idx, 
        d_values);
    // Allocate memory for total forces and displacements
    double2 *totalf;
    cudaMalloc(&totalf, N * sizeof(double2));
    double *totalf_flat;
    cudaMalloc(&totalf_flat, 2 * N * sizeof(double));
    double *displacements_flat;
    cudaMalloc(&displacements_flat, 2 * N * sizeof(double));
    // calculate the total forces vector
    calculateForces_kernel<<<nBlocks, blockSize>>>(
        forces,
        motility,
        cellDirectors,
        velocities,
        totalf,
        N
    );
    // Flatten the total forces vector
    double2ToDouble_kernel<<<nBlocks, blockSize>>>(totalf, totalf_flat, N);
    // solve the linear system Ax=b using the cusolver
    cusolverSpHandle_t handle;
    cusolverSpCreate(&handle);
    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT);
    cusparseSetMatFillMode(descrA, CUSPARSE_FILL_MODE_LOWER);
    int singularity = 0;
    cusolverSpDcsrlsvchol(
        handle, 
        N,
        nnz, 
        descrA,
        d_values,
        d_row_ptr,
        d_col_idx,
        totalf_flat, 
        1e-12, // tolerance
        0, // reorder
        displacements_flat, // output displacement_flat
        &singularity);
    if (singularity != 0) {
        fprintf(stderr, "Matrix is singular or ill-conditioned.\n");
        cusolverSpDestroy(handle);
        cusparseDestroyMatDescr(descrA);
        cudaFree(d_row_ptr);
        cudaFree(d_col_idx);
        cudaFree(d_values);
        cudaFree(totalf);
        cudaFree(totalf_flat);
        cudaFree(displacements_flat);
        return false;
    }
    // Destroy the cusolver and cusparse handles
    cusolverSpDestroy(handle);
    cusparseDestroyMatDescr(descrA);
    // Convert the flat displacements back to double2 format
    doubleToDouble2_kernel<<<nBlocks, blockSize>>>(
        displacements_flat,
        displacements,
        N
    );
    // integration and update
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
    
    // Free the allocated memory
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_values);
    cudaFree(totalf);
    cudaFree(totalf_flat);
    cudaFree(displacements_flat);
    return true;
}
