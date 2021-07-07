cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
cmake_policy(SET CMP0074 NEW)
project(BenchMATMUL)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -g")

#find_package(OpenMP)
#if (OpenMP_CXX_FOUND)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    if (USE_MKL)
#        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
#    endif ()
#    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#    message("-- OpenMP detected ---")
#else ()
#    message(WARNING "Building Kronmult without OpenMP support.")
#endif ()

set(BLA_VENDOR Intel10_64lp)
find_package(MKL)
find_package(BLAS)
if (MKL_FOUND)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MKL_LINKER_FLAGS}")
    find_package(MKL REQUIRED)
    include_directories(${MKL_INCLUDE_DIRS})
    link_directories(${MKL_LIBRARIES})
    message("-- BLAS detected ---")
else()
    message(WARNING "Building Kronmult without BLAS support.")
endif()

add_executable(matmul_bench main.cpp)
target_link_libraries(matmul_bench
mkl_intel_lp64
mkl_sequential
#mkl_intel_thread
mkl_core
)
#target_link_libraries(matmul_bench PUBLIC mkl)
