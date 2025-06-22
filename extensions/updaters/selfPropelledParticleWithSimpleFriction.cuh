#ifndef SELF_PROPELLED_PARTICLE_WITH_SIMPLE_FRICTION_CUH
#define SELF_PROPELLED_PARTICLE_WITH_SIMPLE_FRICTION_CUH

#include "selfPropelledParticleWithSimpleFriction.h"


int gpu_computeRowPtr(
                    const int* d_nn,
                    int N,
                    int* d_row_ptr);
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
                    int *d_col_ind,
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
                    int* d_neigh_change);
#endif // SELF_PROPELLED_PARTICLE_WITH_SIMPLE_FRICTION_CUH