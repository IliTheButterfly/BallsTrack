# This cmakelists is derived from https://github.com/evlasblom/cuda-opencv-examples/blob/68a9b158833c6a8bf4ddf2f7ba00784b74d2326b/CMakeLists.txt

# compiler options, set c++11 (requires cmake 3.2 or higher)
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)

project(ballstrack)

set(CMAKE_BUILD_TYPE Release)

###############################################################################
## find libraries #############################################################
###############################################################################

# FIND LIBRARIES
find_package(OpenCV REQUIRED)
# find_package(CUDA REQUIRED)
# find_package(Boost 1.40 COMPONENTS program_options REQUIRED)

# INCLUDE HEADERS
include_directories(${OpenCV_INCLUDE_DIRS})
# include_directories(${Boost_INCLUDE_DIR})
# include_directories(include)

###############################################################################
## target definitions #########################################################
###############################################################################

# COMPILER FLAGS AND DEFINITIONS
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_30,code=sm_30)
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11)

add_executable(main src/main.cpp)

# target_link_libraries(main ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_link_libraries(main ${OpenCV_LIBS})
