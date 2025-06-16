#include "std_include.h"
#define ENABLE_CUDA
#include <stdio.h>
#include "Simulation.h"
#include "voronoiQuadraticEnergy.h"
#include "selfPropelledParticleWithSimpleFriction.h"
#include "simpleVoronoiDatabase.h"
#include "logEquilibrationStateWriter.h"   //writes out equilibration state
// #include <iostream>
// #include <fstream>


/*!
This file compiles to produce an executable that can be used to reproduce the timing information
in the main cellGPU paper. It sets up a simulation that takes control of a voronoi model and a simple
model of active motility
NOTE that in the output, the forces and the positions are not, by default, synchronized! The NcFile
records the force from the last time "computeForces()" was called, and generally the equations of motion will 
move the positions. If you want the forces and the positions to be sync'ed, you should call the
Voronoi model's computeForces() funciton right before saving a state.
*/
int main(int argc, char*argv[])
{
    //...some default parameters
    int numpts = 10000; //number of cells
    int USE_GPU = 0; //0 or greater uses a gpu, any negative number runs on the cpu
    int c;
    int tSteps = 10000*100; //number of time steps to run after initialization
    int initSteps = 100*100; //number of initialization steps
    int idx = 0;   //repeated ensemble no. 

    double dt = 0.01; //the time step size
    double p0 = 3.7;  //the preferred perimeter
    double a0 = 1.0;  // the preferred area
    double v0 = 0.1;  // the velocity
    double Dr = 1;    // the diffusivity
    //The defaults can be overridden from the command line
    while((c=getopt(argc,argv,"n:g:m:s:r:a:i:v:b:x:y:z:p:t:e:")) != -1)
        switch(c)
        {
            case 'n': numpts = atoi(optarg); break;
            case 't': tSteps = atoi(optarg); break;
            case 'g': USE_GPU = atoi(optarg); break;
            case 'i': initSteps = atoi(optarg); break;
            case 'e': dt = atof(optarg); break;
            case 'p': p0 = atof(optarg); break;
            case 'a': a0 = atof(optarg); break;
            case 'v': v0 = atof(optarg); break;
            case 'r': Dr = atof(optarg); break;
            case 'm': idx = atof(optarg); break;
            case '?':
                    if(optopt=='c')
                        std::cerr<<"Option -" << optopt << "requires an argument.\n";
                    else if(isprint(optopt))
                        std::cerr<<"Unknown option '-" << optopt << "'.\n";
                    else
                        std::cerr << "Unknown option character.\n";
                    return 1;
            default:
                       abort();
        };

    clock_t t1,t2; //clocks for timing information
    bool reproducible = false; // if you want random numbers with a more random seed each run, set this to false
    //check to see if we should run on a GPU
    bool initializeGPU = true;
    bool gpu = chooseGPU(USE_GPU);
    if (!gpu)
        initializeGPU = false;

    //set-up a log-spaced state saver...can add as few as 1 database, or as many as you'd like. "0.1" will save 10 states per decade of time
    logEquilibrationStateWriter lewriter(0.1);
    char dataname[256];
    double equilibrationTime = dt*initSteps;
    vector<long long int> offsets;
    offsets.push_back(1);
    //offsets.push_back(100);offsets.push_back(1000);offsets.push_back(50);
    for(int ii = 0; ii < offsets.size(); ++ii)
        {
        sprintf(dataname,"Rawdata_N%i_p%.5f_v%.5f_Dr%.5f_et%.6f_idx%i.nc",numpts,p0,v0,Dr,offsets[ii]*dt,idx);
        shared_ptr<simpleVoronoiDatabase> ncdat=make_shared<simpleVoronoiDatabase>(numpts,dataname,fileMode::replace);
        lewriter.addDatabase(ncdat,offsets[ii]);
        }
    lewriter.identifyNextFrame();


    cout << "initializing a system of " << numpts << " cells at velocity " << v0 << endl;
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

    //run for a few initialization timesteps
    printf("starting initialization\n");
    for(long long int ii = 0; ii < initSteps; ++ii)
        {
        sim->performTimestep();
        };
    voronoiModel->computeGeometry();
    printf("Finished with initialization\n");
    cout << "current q = " << voronoiModel->reportq() << endl;
    //the reporting of the force should yield a number that is numerically close to zero.
    voronoiModel->reportMeanCellForce(false);

    //run for additional timesteps, compute dynamical features, and record timing information
    FILE * pFile;
    for(long long int ii = 0; ii <= tSteps; ++ii)
        {
        if (ii == lewriter.nextFrameToSave)
            {
            lewriter.writeState(voronoiModel,ii);
//            CR_MSD = dynFeat.computeCageRelativeMSD(voronoiModel->returnPositions());
//            Overlap = dynFeat.computeOverlapFunction(voronoiModel->returnPositions());
//            CR_Fs_Chi4=dynFeat.computeCageRelativeFsChi4(voronoiModel->returnPositions());
//            pFile=fopen(dataname2,"a");
//            fprintf(pFile,"%lld ,%f, %f, %f, %f\n",ii, CR_MSD, Overlap, CR_Fs_Chi4.x, CR_Fs_Chi4.y);
//            fclose(pFile);
            cout << "Finishing step: " << ii << "Next Frame to save is: " << lewriter.nextFrameToSave<< endl;
            }
        sim->performTimestep();
        };
    if(initializeGPU)
        cudaDeviceReset();
    return 0;
};
