#include "selfPropelledParticleWithSimpleFriction.h"
#include "selfPropelledParticleWithSimpleFriction.cuh"
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>

/*! \file selfPropelledParticleWithSimpleFriction.cpp */
selfPropelledParticleWithSimpleFriction::selfPropelledParticleWithSimpleFriction(int _N, double _xi_rel, bool _useGPU) 
        : selfPropelledParticleDynamics{_N, _useGPU}
        , xi_rel{_xi_rel}
    {
        int maxRows = 2 * _N;
        int maxNnz = 7 * maxRows; // Maximum non-zero entries in the friction matrix
        // Allocate memory for the sparse matrix representation on the device
        cudaMalloc(&d_row_ptr, (maxRows + 1) * sizeof(int));
        cudaMemset(d_row_ptr, 0, sizeof(int));
        cudaMalloc(&d_col_idx, maxNnz * sizeof(int));
        cudaMalloc(&d_values, maxNnz * sizeof(double));
        cudaMalloc(&velocity_flat, maxRows * sizeof(double));
        cudaMalloc(&totalf_flat,  maxRows * sizeof(double));
        cudaMalloc(&d_row_sizes,  maxRows * sizeof(int));
        // create and initialize cuSolver/cuSparse handles
        cusolverStatus_t cusolver_status = cusolverSpCreate(&handle);
        cusparseCreateMatDescr(&descrA);
        cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
        ArrayHandle<int> h_nn(activeModel->neighborNum, access_location::host, access_mode::read);
        ArrayHandle<int> h_n(activeModel->neighbors, access_location::host, access_mode::read);
        old_nn.assign(h_nn.data, h_nn.data + activeModel->neighborNum.getNumElements());
        old_neigh.assign(h_n.data, h_n.data + activeModel->neighbors.getNumElements());
        Index2D n_idx = activeModel->n_idx;
        gpu_symbolic_solver_phase(h_nn.data,
                                  h_n.data,
                                  d_row_ptr,
                                  d_col_idx,
                                  d_values,
                                  d_row_sizes,
                                  n_idx,
                                  _N, 
                                  xi_rel, 
                                  handle, 
                                  descrA
        );
        
    } 


void selfPropelledParticleWithSimpleFriction::computeFrictionMatrix(Eigen::SparseMatrix<double> &mat)
    {
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve(14*Ndof);

    ArrayHandle<int> h_nn(activeModel->neighborNum,access_location::host,access_mode::read);
    ArrayHandle<int> h_n(activeModel->neighbors,access_location::host,access_mode::read);
    ArrayHandle<double2> h_p(activeModel->cellPositions,access_location::host,access_mode::read);
    for (int i = 0; i < Ndof; ++i)
        {
        for (int j = 0; j < h_nn.data[i]; ++j)
            {
            int k = h_n.data[activeModel->n_idx(j,i)];
            tripletList.push_back(T(2*i, 2*k, -xi_rel));
            tripletList.push_back(T(2*i+1, 2*k+1, -xi_rel));
            }
        tripletList.push_back(T(2*i, 2*i, xi_sub + xi_rel*h_nn.data[i]));
        tripletList.push_back(T(2*i+1, 2*i+1, xi_sub + xi_rel*h_nn.data[i]));
        }
    mat.setFromTriplets(tripletList.begin(), tripletList.end());
    }

/*!
The straightforward CPU implementation
*/
void selfPropelledParticleWithSimpleFriction::integrateEquationsOfMotionCPU()
    {
    Eigen::SparseMatrix<double> frictionMatrix(2*Ndof, 2*Ndof);
    computeFrictionMatrix(frictionMatrix);
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(frictionMatrix);
    if (solver.info() != Eigen::Success)
        {
        throw std::runtime_error("Matrix decomposition failed in selfPropelledParticleWithSimpleFriction.");
        }
    Eigen::VectorXd forces(2*Ndof), v(2*Ndof);
    activeModel->computeForces();

    {// scope for array handles
        ArrayHandle<double2> h_f(activeModel->returnForces(),access_location::host,access_mode::read);
        ArrayHandle<double> h_cd(activeModel->cellDirectors);
        ArrayHandle<double2> h_v(activeModel->cellVelocities);
        ArrayHandle<double2> h_disp(displacements,access_location::host,access_mode::overwrite);
        ArrayHandle<double2> h_motility(activeModel->Motility,access_location::host,access_mode::read);

        for (int ii = 0; ii < Ndof; ++ii)
            {
            //displace according to current velocities and forces
            double f0i = h_motility.data[ii].x;
            h_v.data[ii].x =  f0i * cos(h_cd.data[ii]);
            h_v.data[ii].y =  f0i * sin(h_cd.data[ii]);
            double2 Vcur = h_v.data[ii];
            forces(2*ii) = Vcur.x + h_f.data[ii].x;
            forces(2*ii+1) = Vcur.y + h_f.data[ii].y;
            }
        v = solver.solve(forces);
        if (solver.info() != Eigen::Success)
            {
            throw std::runtime_error("Matrix solve failed in selfPropelledParticleWithSimpleFriction.");
            }
        for (int ii = 0; ii < Ndof; ++ii)
            { // displace according to current velocities
            h_disp.data[ii].x = deltaT*v(2*ii);
            h_disp.data[ii].y = deltaT*v(2*ii+1);
            // rotate the velocity vector a bit
            double Dri = h_motility.data[ii].y;
            double theta = h_cd.data[ii];
            //rotate the velocity vector a bit
            double randomNumber = noise.getRealNormal();
            h_cd.data[ii] = theta + randomNumber*sqrt(2.0*deltaT*Dri);
            };
    }// end array handle scoping
    activeModel->moveDegreesOfFreedom(displacements);
    activeModel->enforceTopology();
}
/*!
The GPU implementation of the self-propelled particle dynamics with simple friction
*/
void selfPropelledParticleWithSimpleFriction::integrateEquationsOfMotionGPU()
    {
         activeModel->computeForces();
        {//scope for array Handles
        ArrayHandle<double2> d_f(activeModel->returnForces(),access_location::device,access_mode::read);
        ArrayHandle<double> d_cd(activeModel->cellDirectors,access_location::device,access_mode::readwrite);
        ArrayHandle<double2> d_v(activeModel->cellVelocities,access_location::device,access_mode::readwrite);
        ArrayHandle<double2> d_disp(displacements,access_location::device,access_mode::overwrite);
        ArrayHandle<double2> d_motility(activeModel->Motility,access_location::device,access_mode::read);
        ArrayHandle<curandState> d_RNG(noise.RNGs,access_location::device,access_mode::readwrite);
        ArrayHandle<int> d_nn(activeModel->neighborNum,access_location::device,access_mode::read);
        ArrayHandle<int> d_n(activeModel->neighbors,access_location::device,access_mode::read);
        ArrayHandle<int> h_nn(activeModel->neighborNum,access_location::host,access_mode::read);
        ArrayHandle<int> h_n(activeModel->neighbors,access_location::host,access_mode::read);
        Index2D n_idx = activeModel->n_idx;
        if (std::equal(old_nn.begin(), old_nn.end(), h_nn.data) ||
            std::equal(old_neigh.begin(), old_neigh.end(), h_n.data))
            {
            // No need to recompute the friction matrix if neighbors haven't changed
            }
        else
            {
            // Recompute the friction matrix if neighbors have changed
            gpu_symbolic_solver_phase(d_nn.data,
                                      d_n.data,
                                      d_row_ptr,
                                      d_col_idx,
                                      d_values,
                                      d_row_sizes,
                                      n_idx,
                                      Ndof, 
                                      xi_rel, 
                                      handle, 
                                      descrA);
            old_nn.assign(h_nn.data, h_nn.data + activeModel->neighborNum.getNumElements());
            old_neigh.assign(h_n.data, h_n.data + activeModel->neighbors.getNumElements());
            }
        gpu_spp_friction_eom_integration(
                    d_nn.data,
                    d_n.data,
                    d_f.data,
                    d_v.data,
                    velocity_flat,
                    totalf_flat,
                    d_disp.data,
                    d_motility.data,
                    d_cd.data,
                    d_row_ptr,
                    d_col_idx,
                    d_values,
                    d_row_sizes,
                    d_RNG.data,
                    Ndof,
                    n_idx,
                    deltaT,
                    Timestep,
                    xi_sub,
                    xi_rel,
                    handle,
                    descrA);
        };//end array handle scope
    activeModel->moveDegreesOfFreedom(displacements);
    activeModel->enforceTopology();
    }

