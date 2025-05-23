if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    add_definitions(-DDEBUGFLAGUP)
endif(${CMAKE_BUILD_TYPE} MATCHES "Debug")

include_directories(
            ${EIGEN3_INCLUDE_DIR}
            ${CUDA_INCLUDE_DIRS}
            ${CMAKE_SOURCE_DIR}/inc 
            ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_SOURCE_DIR}/src/utility
            ${CMAKE_SOURCE_DIR}/src/analysis
            ${CMAKE_SOURCE_DIR}/src/models
            ${CMAKE_SOURCE_DIR}/src/forces
            ${CMAKE_SOURCE_DIR}/src/updaters
            ${CMAKE_SOURCE_DIR}/src/simulation
        )

add_library(utility
    cellListGPU.cpp
    eigenMatrixInterface.cpp
    hilbert_curve.cpp
    noiseSource.cpp
    )
add_library(utilityGPU
    cellListGPU.cu
    noiseSource.cu
    utilities.cu
    )
set_target_properties(utilityGPU PROPERTIES
                            CUDA_SEPARABLE_COMPILATION ON)
