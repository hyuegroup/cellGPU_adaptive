#ifndef SELF_PROPELLED_PARTICLE_WITH_SIMPLE_FRICTION_H
#define SELF_PROPELLED_PARTICLE_WITH_SIMPLE_FRICTION_H

#include "selfPropelledParticleDynamics.h"

#include <Eigen/SparseCore>

class selfPropelledParticleWithSimpleFriction : public selfPropelledParticleDynamics
{
public:
    // Constructor
    selfPropelledParticleWithSimpleFriction(int _N, double _xi_rel) : selfPropelledParticleDynamics(_N, false)
    {
        xi_rel = _xi_rel;
    }
    // Add public methods and members here
    virtual void integrateEquationsOfMotionCPU(); 
    virtual void integrateEquationsOfMotionGPU();
    void setXiSub(double _xi_sub){ xi_sub = _xi_sub;};
    void setXiRel(double _xi_rel){ xi_rel = _xi_rel;};
protected:
    // Add protected members here
    virtual void computeFrictionMatrix(Eigen::SparseMatrix<double> &mat);
    double xi_rel; // Relative friction coefficient
    double xi_sub = 1.; // Substrate Friction coefficient
};

#endif // SELF_PROPELLED_PARTICLE_WITH_SIMPLE_FRICTION_H