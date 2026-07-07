cmake_minimum_required(VERSION 3.26)

project(unittest LANGUAGES CXX VERSION 2.0.0 DESCRIPTION "libdriveless test")

find_package(driveless REQUIRED)
get_target_property(_driveless_defs driveless::driveless INTERFACE_COMPILE_DEFINITIONS)
if(_driveless_defs MATCHES "DRIVELESS_CUDA_ENABLED")
    message(STATUS "driveless was built with CUDA support")
    set(DRIVELESS_CUDA_ENABLED ON)
else()
    message(WARNING "driveless was built without CUDA support")
    set(DRIVELESS_CUDA_ENABLED OFF)
endif()

if (DRIVELESS_CUDA_ENABLED)
    # Optional: explicitly set NVCC if needed
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc CACHE FILEPATH "Path to nvcc")
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES all-major)
    find_package(CUDAToolkit REQUIRED)
endif()


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES all-major)


# Optional: explicitly set NVCC if needed
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc CACHE FILEPATH "Path to nvcc")

include(GNUInstallDirs)

find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(GSL REQUIRED)

if (DRIVELESS_CUDA_ENABLED)
    find_package(CUDAToolkit REQUIRED)
endif()
find_package(driveless REQUIRED)

#find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
    message(STATUS "CUDA Toolkit version is >= 13.0 (found ${CUDAToolkit_VERSION})")
endif()

get_target_property(_driveless_defs driveless::driveless INTERFACE_COMPILE_DEFINITIONS)
if(_driveless_defs MATCHES "DRIVELESS_CUDA_ENABLED")
    message(WARNING "testing FastRRT for CUDA support")
else()
    #message(FATAL_ERROR "FastRRT currently requires libdriveless to be compiled with CUDA support")
    message(WARNING "testing FastRRT for CPU")
endif()


if (DRIVELESS_CUDA_ENABLED)
    string(REPLACE "." ";" CUDA_VER_LIST ${CMAKE_CUDA_COMPILER_VERSION})
    list(GET CUDA_VER_LIST 0 CUDA_MAJOR)
    list(GET CUDA_VER_LIST 1 CUDA_MINOR)

    add_compile_definitions(CUDA_VERSION_MAJOR=${CUDA_MAJOR})
    add_compile_definitions(CUDA_VERSION_MINOR=${CUDA_MINOR})
endif()

option(CUDA_LIB_JETSON "true" OFF)

if(CUDA_LIB_JETSON)
    find_package(jetson-utils REQUIRED)
endif()

# Detect architecture
execute_process(
    COMMAND uname -m
    COMMAND tr -d '\n'
    OUTPUT_VARIABLE ARCHITECTURE
)
message(STATUS "Architecture: ${ARCHITECTURE}")

pkg_check_modules(deps REQUIRED IMPORTED_TARGET glib-2.0)
pkg_search_module(GLIB REQUIRED glib-2.0)

# Locate GTest
find_package(GTest REQUIRED)
find_package(driveless REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${CUDAToolkit_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})   # ✅ Now works

link_directories(/usr/local/lib/driveless)

SET(TARGET_H
    # (headers if any)
)

if (DRIVELESS_CUDA_ENABLED)
add_executable(
  ${CMAKE_PROJECT_NAME} 
    ../src/cuda/kinematics.cu
    ../src/cuda/path_feasible.cu
    ../src/cuda/random_number_gen.cu
    ../src/cuda/graph.cu
    ../src/cuda/graph_list.cu
    ../src/cuda/graph_best_node.cu
    ../src/cuda/graph_optimization.cu
    ../src/cuda/graph_expansion.cu
    ../src/cuda/graph_nodes_in_range.cu
    ../src/cuda/graph_costs.cu
    ../src/cuda/graph_clear.cu
    ../src/cuda/apf.cu
    ../src/cuda/control_expansion.cu
    ../src/cuda/graph_expansion_search_zones.cu
    ../src/cuda/graph_collision.cu
    ../src/cuda/graph_hermite_connection.cu
    ../src/cuda/graph_direct_goal_connection.cu
    ../src/cuda/graph_node_expansion.cu
    ../src/fastrrt.cpp
    ../src/python_bind.cpp
    ../src/python_cuda_graph_bind.cpp
    ../src/interpolation.cpp
    ../src/graph_check_dag_consistency.cpp

    src/test_utils.cpp
    src/unittests.cpp
    src/test_apf.cpp
    src/test_graph.cpp
    src/test_graph_node_derivation.cpp
    src/test_kinematics.cpp
    src/test_graph_best_node.cpp
    src/test_graph_can_connect_hermite.cpp
    src/test_graph_dag_consistency.cpp
    src/test_graph_solve_collisions.cpp
    #src/test_graph_direct_goal_connection.cpp
    src/test_graph_dump.cpp
    src/test_rrt.cpp
)
else()
    set_source_files_properties(
        ../src/cuda/graph.cu
        ../src/cuda/graph_direct_goal_connection.cu
        ../src/cuda/graph_optimization.cu
        ../src/cuda/graph_node_expansion.cu
        ../src/cuda/control_expansion.cu        
        PROPERTIES LANGUAGE CXX)


  add_executable(
    ${CMAKE_PROJECT_NAME} 
    ../src/cpu/kinematics.cpp
    ../src/cpu/path_feasible.cpp
    ../src/cpu/random_number_gen.cpp
    ../src/cuda/graph.cu
    ../src/cuda/control_expansion.cu    
    ../src/cpu/graph_list.cpp
    ../src/cpu/graph_best_node.cpp
    ../src/cpu/graph_expansion.cpp
    ../src/cpu/graph_nodes_in_range.cpp
    ../src/cpu/graph_costs.cpp
    ../src/cpu/graph_clear.cpp
    ../src/cpu/apf.cpp
    ../src/cpu/graph_expansion_search_zones.cpp
    ../src/cpu/graph_collision.cpp
    ../src/cpu/graph_hermite_connection.cpp
    ../src/fastrrt.cpp
    ../src/python_bind.cpp
    ../src/python_cuda_graph_bind.cpp
    ../src/interpolation.cpp
    ../src/graph_check_dag_consistency.cpp
    ../src/cuda/graph_direct_goal_connection.cu
    ../src/cuda/graph_optimization.cu
    ../src/cuda/graph_node_expansion.cu
    src/test_utils.cpp
    src/unittests.cpp
    src/test_apf.cpp
    src/test_graph.cpp
    src/test_graph_node_derivation.cpp
    src/test_kinematics.cpp
    src/test_graph_best_node.cpp
    src/test_graph_can_connect_hermite.cpp
    src/test_graph_dag_consistency.cpp
    src/test_graph_solve_collisions.cpp
    #src/test_graph_direct_goal_connection.cpp
    src/test_graph_dump.cpp
    src/test_rrt.cpp
  )
endif()

set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

if(THREADS_HAVE_PTHREAD_ARG)
  target_compile_options(${CMAKE_PROJECT_NAME} PUBLIC "-pthread")
endif()
if(CMAKE_THREAD_LIBS_INIT)
  target_link_libraries(${CMAKE_PROJECT_NAME} "${CMAKE_THREAD_LIBS_INIT}")
endif()

target_link_libraries( 
  ${CMAKE_PROJECT_NAME}  
  PkgConfig::deps 
  ${GTEST_LIBRARIES} 
  ${CUDA_LIBRARIES}
  ${CUDA_FRAME}
  ${OpenCV_LIBS}          # ✅ Now properly resolved
  GSL::gsl 
  GSL::gslcblas
  driveless
)



# # link my-recognition to jetson-inference library
# target_link_libraries( 
#   ${CMAKE_PROJECT_NAME}  
#   PkgConfig::deps 
#   ${GTEST_LIBRARIES} 
#   ${CUDA_LIBRARIES}
#   ${CUDA_FRAME}
#   ${OpenCV_LIBS}
#   ${CUDA_FRAME}
#   # ${Python3_LIBRARIES}
#   ${LIB_CUDA_UTILS}

# )

