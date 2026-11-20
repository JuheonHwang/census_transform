cmake_minimum_required(VERSION 3.18)

# GPU architecture: auto-detect with CMake >= 3.24, otherwise a common fallback list.
# Override with -DCMAKE_CUDA_ARCHITECTURES=86 (for example).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 61 75 86)
  endif()
endif()

project(census_stereo_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs highgui)

# One executable per optimisation stage. main.cpp and pfm_io.cpp are shared; exactly one
# .cu file (which defines DiffBoxImages) is linked into each executable.
function(add_census_executable target cuda_source)
  add_executable(${target} src/main.cpp src/pfm_io.cpp ${cuda_source})
  target_include_directories(${target} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(${target} PRIVATE CUDA::cudart ${OpenCV_LIBS})
endfunction()

add_census_executable(census_v1 src/v1_baseline.cu)   # separate XOR / Sum kernels, full cost volume
add_census_executable(census_v2 src/v2_fused.cu)      # fused XOR + Sum, full cost volume
add_census_executable(census_v3 src/v3_rolling.cu)    # fused + rolling cost buffers (memory independent of max_disp)
