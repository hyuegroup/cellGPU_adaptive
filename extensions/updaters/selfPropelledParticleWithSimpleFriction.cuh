#ifndef SELF_PROPELLED_PARTICLE_WITH_SIMPLE_FRICTION_CUH
#define SELF_PROPELLED_PARTICLE_WITH_SIMPLE_FRICTION_CUH

#include "selfPropelledParticleWithSimpleFriction.h"


int gpu_computeRowPtr(
                    const int* d_nn,
                    int N,
                    int* d_row_ptr);
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
                    cusolverSpHandle_t handle,
                    cusparseMatDescr_t descrA);
#endif // SELF_PROPELLED_PARTICLE_WITH_SIMPLE_FRICTION_CUH