#include "std_include.h"
#define ENABLE_CUDA
#include <stdio.h>
#include "Simulation.h"
#include "voronoiQuadraticEnergy.h"
#include "selfPropelledParticleWithSimpleFriction.h"
#include "simpleVoronoiDatabase.h"
#include <chrono>
#include <iostream>
#include <fstream>


/*!
This file compiles to produce an executable that can be used to reproduce the timing information
in the main cellGPU paper. It sets up a simulation that takes control of a voronoi model and a simple
model of active motility
NOTE that in the output, the forces and the positions are not, by default, synchronized! The NcFile
records the force from the last time "computeForces()" was called, and generally the equations of motion will 
move the positions. If you want the forces and the positions to be sync'ed, you should call the
Voronoi model's computeForces() funciton right before saving a state.
*/
double benchmark(int N, int use_gpu)
{
    //...some default parameters
    int numpts = N; //number of cells
    int USE_GPU = use_gpu; //0 or greater uses a gpu, any negative number runs on the cpu
    int c;
    int tSteps = 1000; //number of time steps to run after initialization
    int idx = 0;   //repeated ensemble no. 

    double dt = 0.01; //the time step size
    double p0 = 3.7;  //the preferred perimeter
    double a0 = 1.0;  // the preferred area
    double v0 = 0.01;  // the velocity
    double Dr = 1;    // the diffusivity

    clock_t t1,t2; //clocks for timing information
    bool reproducible = false; // if you want random numbers with a more random seed each run, set this to false
    //check to see if we should run on a GPU
    bool initializeGPU = true;
    bool gpu = chooseGPU(USE_GPU);
    if (!gpu)
        initializeGPU = false;
    shared_ptr<selfPropelledParticleWithSimpleFriction> spp = make_shared<selfPropelledParticleWithSimpleFriction>(numpts,1.0);

    //define a voronoi configuration with a quadratic energy functional
    shared_ptr<VoronoiQuadraticEnergy> voronoiModel  = make_shared<VoronoiQuadraticEnergy>(numpts,1.0,4.0,reproducible,initializeGPU);

    //set the cell preferences to uniformly have A_0 = 1, P_0 = p_0
    voronoiModel->setCellPreferencesWithRandomAreas(p0,0.8,1.2);
    voronoiModel->setv0Dr(v0,Dr);

    //combine the equation of motion and the cell configuration in a "Simulation"
    SimulationPtr sim = make_shared<Simulation>();
    sim->setConfiguration(voronoiModel);
    sim->addUpdater(spp,voronoiModel);
    //set the time step size
    sim->setIntegrationTimestep(dt);
    //initialize Hilbert-curve sorting... can be turned off by commenting out this line or seting the argument to a negative number
    //sim->setSortPeriod(initSteps/10);
    //set appropriate CPU and GPU flags
    sim->setCPUOperation(!initializeGPU);
    if (!gpu)
        sim->setOmpThreads(abs(USE_GPU));
    sim->setReproducible(reproducible);

    //run for additional timesteps, compute dynamical features, and record timing information
    auto start = std::chrono::high_resolution_clock::now();
    for(long long int ii = 0; ii <= tSteps; ++ii)
        {
        sim->performTimestep();
        };
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    if(initializeGPU)
        cudaDeviceReset();
    return elapsed_ms / tSteps; //return the average time per step in milliseconds
};

int main()
{
    std::vector<int> N{2000,2000,4000,8000,16000};
    std::vector<double> timePerStep_cpu(5);
    std::vector<double> timePerStep_gpu(5);
    for (int i= 0; i < 5; ++i)
    {
        timePerStep_cpu[i]=benchmark(N[i],-1); //run on the CPU
        timePerStep_gpu[i]=benchmark(N[i],0);  //run on the GPU
        cudaDeviceReset();
        cout << "N = " << N[i] << ", CPU time per step = " << timePerStep_cpu[i] << " ms, GPU time per step = " << timePerStep_gpu[i] << " ms" << endl;
    }
    ofstream File("timePerStep.txt");
    if (!File.is_open())
    {
        cerr << "Error opening file for writing!" << endl;
        return 1;
    }  
    for (int i = 0; i < 5; ++i)
    {
        File << N[i] << "\t" << timePerStep_cpu[i] << "\t" << timePerStep_gpu[i] << endl;
    }
    File.close();
    return 1;
}
