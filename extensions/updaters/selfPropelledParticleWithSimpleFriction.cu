#include "std_include.h"
#include "curand_kernel.h"
#include "indexer.h"
#include "selfPropelledParticleWithSimpleFriction.cuh"

__global__ void checkNeighborChange_kernel(
    int* old_nn,
    int* old_n,
    const int* __restrict__ new_nn,
    const int* __restrict__ new_n,
    Index2D n_idx,
    int N,
    int* changed)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    if (old_nn[idx] != new_nn[idx]) {
        atomicExch(changed, 1);
        //*changed = 1;
        old_nn[idx] = new_nn[idx];
        // return;
    }

    int nn = new_nn[idx];
    for (int k = 0; k < nn; ++k) {
        if (old_n[n_idx(k, idx)] != new_n[n_idx(k, idx)]) {
            atomicExch(changed, 1);
            //*changed = 1;
            old_n[n_idx(k, idx)] = new_n[n_idx(k, idx)];
            // return;
        }
    }
}

__global__ void calculateForces_kernel(
    const double2 * __restrict__ forces,
    const double2 * __restrict__ motility,
    const double * __restrict__ cellDirectors,
    double *totalf_flat,
    int N)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    // Calculate the velocity vector based on motility and cell director
    double v0 = motility[idx].x;
    double cosValue;
    double sinValue;
    sincos(cellDirectors[idx],&sinValue,&cosValue);
    totalf_flat[2*idx] = v0 * cosValue + forces[idx].x;
    totalf_flat[2*idx+1] = v0 * sinValue + forces[idx].y;
}

__global__ void fillRowSize_kernel(int *row_sizes, const int * __restrict__ d_nn, int N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2*N) return;
    int cellidx = i / 2; // Determine the cell index
    row_sizes[i] = d_nn[cellidx] + 1; // both x and y direction
}


__global__ void buildFrictionMatrixCSR_kernel(
    const int* __restrict__ neighborNum,
    const int* __restrict__ neighbors,
    Index2D n_idx,
    int N,
    double xi_sub,
    double xi_rel,
    int* row_ptr,
    int* col_idx,
    double* values)
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
                                           const double2 * __restrict__ motility,
                                           double *cellDirectors,
                                           double2 *velocities,
                                           const double * __restrict__ velocity_flat,
                                           curandState *RNGs,
                                           int N,
                                           double deltaT,
                                           int Timestep)
    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >=N)
        return;

    velocities[idx].x = velocity_flat[2*idx];
    velocities[idx].y = velocity_flat[2*idx + 1];
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
    if (tempTheta > M_PI)   // Normalize to [-pi, pi]
        tempTheta-=2*M_PI; 
    else if (tempTheta < M_PI) 
        tempTheta+=2*M_PI; 
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
    int blockSize = 256;
    int nBlocks = (nrows + blockSize - 1) / blockSize;
    fillRowSize_kernel<<<nBlocks, blockSize>>>(d_row_sizes, d_nn, N);
    // Do inclusive scan using Thrust to get row_ptr
    thrust::inclusive_scan(thrust::device, d_row_sizes, d_row_sizes + nrows, d_row_ptr+1);
    int nnz = thrust::reduce(thrust::device, d_row_sizes, d_row_sizes + nrows, 0, thrust::plus<int>());
    return nnz; // Return the number of non-zero entries
}

bool gpu_spp_friction_eom_integration(   
                    int* new_nn,
                    int* new_n,
                    int* old_nn,
                    int* old_n,
                    int nnz,
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
                    cudssHandle_t handle,
                    cudssConfig_t config,
                    cudssData_t data,
                    cudssMatrix_t A,
                    cudssMatrix_t b,
                    cudssMatrix_t x,
                    int* d_neigh_change)
{
    int blockSize = 128;
    int nBlocks = (N + blockSize - 1) / blockSize;
    // check neighbor change
    cudaMemset(d_neigh_change, 0, sizeof(int));
    checkNeighborChange_kernel<<<nBlocks,blockSize>>>(old_nn,old_n,new_nn,new_n,n_idx,N,d_neigh_change);
    int h_neigh_change = 0;
    cudaMemcpy(&h_neigh_change, d_neigh_change, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_neigh_change==0)
        {
        // No need to recompute the friction matrix and redo the symbolic analysis if neighbors haven't changed
        }
    else
        {
        int nnz_check = gpu_computeRowPtr(new_nn, N, d_row_ptr,d_row_sizes);
        if (nnz_check != nnz)
        {
            cout << "nnz changes from " << nnz << "to" << nnz_check <<endl;
            nnz = nnz_check;
            //resize the related things
            cudaMalloc(&d_col_idx, nnz * sizeof(int));
            cudaMalloc(&d_values, nnz * sizeof(double));
            cudssMatrixCreateCsr(
                &A,
                /*rows=*/2*N,
                /*cols=*/2*N,
                /*nnz=*/nnz,
                d_row_ptr,
                NULL,
                d_col_idx,
                d_values,
                CUDA_R_32I,    // index type
                CUDA_R_64F, // value type
                CUDSS_MTYPE_SPD, // Matrix type  symmetric positive definite
                CUDSS_MVIEW_FULL, //Matrix view type
                CUDSS_BASE_ZERO); // value type
        }
        // Recompute the friction matrix if neighbors have changed
        nBlocks = (2*N + blockSize - 1) / blockSize;
        buildFrictionMatrixCSR_kernel<<<nBlocks, blockSize>>>(
            new_nn, 
            new_n, 
            n_idx, 
            N, 
            xi_sub, 
            xi_rel, 
            d_row_ptr, 
            d_col_idx, 
            d_values);
        // Symbolic analysis (pattern only)
        cudssExecute(handle,
             CUDSS_PHASE_ANALYSIS,
             config,
             data,
             A, x, b);
        // update the old_nn and old_n for next step 
        // old_nn.assign(h_nn.data,h_nn.data+N);
        // old_n.assign(h_n.data,h_n.data+neighbors.getNumElements()); 
        }
    // calculate the total forces vector, which automatically update b
        nBlocks = (N + blockSize - 1) / blockSize;
        calculateForces_kernel<<<nBlocks, blockSize>>>(
            forces,
            motility,
            cellDirectors,
            totalf_flat,
            N
        );
    // Numeric factorization
    cudssExecute(handle,
             CUDSS_PHASE_FACTORIZATION,
             config,
             data,
             A, x, b);
    // Solve
    cudssExecute(handle,
             CUDSS_PHASE_SOLVE,
             config,
             data,
             A, x, b);
    // integration and update
    nBlocks = (N + blockSize - 1) / blockSize;
    spp_friction_eom_integration_kernel<<<nBlocks, blockSize>>>(
        displacements,
        motility,
        cellDirectors,
        velocities,
        velocity_flat,
        RNGs,
        N,
        deltaT,
        Timestep
    );
    return true;
}
