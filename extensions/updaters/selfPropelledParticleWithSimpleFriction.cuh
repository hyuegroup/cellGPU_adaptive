#ifndef SELF_PROPELLED_PARTICLE_WITH_SIMPLE_FRICTION_CUH
#define SELF_PROPELLED_PARTICLE_WITH_SIMPLE_FRICTION_CUH

#include "std_include.h"
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>
int gpu_computeRowPtr(
                    const int* d_nn,
                    int N,
                    int* d_row_ptr);
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
                    double xi_rel);
#endif // SELF_PROPELLED_PARTICLE_WITH_SIMPLE_FRICTION_CUH