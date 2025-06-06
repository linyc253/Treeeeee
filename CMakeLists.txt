cmake_minimum_required(VERSION 3.22)
project(Treeeeee C CUDA)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O2")

# Add directories
include_directories(${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/lib)
file(GLOB SRC_C src/*.c lib/*.c)
file(GLOB SRC_CU src/*.cu)

# Define output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

# Combine sources
set(SOURCES ${SRC_C})

option(CUDA "Enable CUDA support" OFF)
if(CUDA)
    list(APPEND SOURCES ${SRC_CU})
endif()

# Add executable
add_executable(treeeeee ${SOURCES})

option(DEBUG "Enable debugging mode" OFF)
if(DEBUG)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DDEBUG -fsanitize=address -g")
endif()

# Enable OpenMP
option(OMP "Enable OpenMP support" OFF)
if(OMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(treeeeee PUBLIC OpenMP::OpenMP_C)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DOMP")
endif()

# Enable CUDA
if(CUDA)
    find_package(CUDA REQUIRED)
    add_definitions(-DCUDA)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O2 -Xcompiler -fPIC -shared -MMD")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DCUDA")
endif()

target_link_libraries(treeeeee PUBLIC m)

# CUDA runtime
if(CUDA)
    target_link_libraries(treeeeee PUBLIC ${CUDA_CUDART_LIBRARY})
endif()

# Output directories
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/build)

