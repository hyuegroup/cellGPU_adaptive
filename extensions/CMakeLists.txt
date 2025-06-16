# extensions/CMakeLists.txt

add_library(my_extensions
   updaters/selfPropelledParticleWithSimpleFriction.cpp
)

target_include_directories(my_extensions
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/updaters
    PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    ../inc
    ../src/utility
    ../src/simulation
    ../src/models
    ../src/updaters
    ../src/analysis
    ../src/databases
)

target_link_libraries(my_extensions
    PRIVATE
    model
    updaters
    analysis
    databases
    simulation
    utility
    modelGPU
    updatersGPU
    utilityGPU
    CGAL::CGAL
    OpenMP::OpenMP_CXX
)
