#ifndef SELF_PROPELLED_PARTICLE_WITH_SIMPLE_FRICTION_H
#define SELF_PROPELLED_PARTICLE_WITH_SIMPLE_FRICTION_H

#include "selfPropelledParticleDynamics.h"
#include "std_include.h"
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <math.h>
#include <Eigen/SparseCore>

class selfPropelledParticleWithSimpleFriction : public selfPropelledParticleDynamics
{
public:
    // Constructor
    selfPropelledParticleWithSimpleFriction(int _N, double _xi_rel = 1.0, bool _useGPU = true);
    ~selfPropelledParticleWithSimpleFriction() 
    {
        if(d_row_ptr) cudaFree(d_row_ptr);
        if(d_col_idx) cudaFree(d_col_idx);
        if(d_values) cudaFree(d_values);
        if(velocity_flat) cudaFree(velocity_flat);
        if(totalf_flat) cudaFree(totalf_flat);
        if(d_row_sizes) cudaFree(d_row_sizes);
        cusparseDestroyMatDescr(descrA);
        cusolverSpDestroy(handle);
    }
    // Add public methods and members here
    virtual void integrateEquationsOfMotionCPU(); 
    virtual void integrateEquationsOfMotionGPU();
    void setXiSub(double _xi_sub){ xi_sub = _xi_sub;};
    void setXiRel(double _xi_rel){ xi_rel = _xi_rel;};
protected:
    // Add protected members here
    virtual void computeFrictionMatrix(Eigen::SparseMatrix<double> &mat);
    int *d_row_sizes = nullptr; // Row sizes for sparse matrix
    int *d_row_ptr = nullptr; // Row pointer for sparse matrix
    int *d_col_idx = nullptr; // Column indices for sparse matrix
    double *d_values = nullptr; // Values for sparse matrix
    double *velocity_flat = nullptr; // Flattened velocity array for GPU
    double *totalf_flat = nullptr; // Flattened total force array for GPU
    double xi_rel; // Relative friction coefficient
    double xi_sub = 1.; // Substrate Friction coefficient
    std::vector<int> old_nn, old_neigh; // Old neighbor numbers and neighbors
    cusolverSpHandle_t handle = nullptr; // cuSolver handle
    cusparseMatDescr_t descrA = nullptr; // cuSparse matrix descriptor
};

#endif // SELF_PROPELLED_PARTICLE_WITH_SIMPLE_FRICTION_H