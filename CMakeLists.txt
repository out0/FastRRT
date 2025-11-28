cmake_minimum_required(VERSION 3.30)

project(fastrrt LANGUAGES CXX CUDA VERSION 2.0.0 DESCRIPTION "libfastrrt")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#set(CMAKE_CUDA_ARCHITECTURES all-major)

# Optional: explicitly set NVCC if needed
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc CACHE FILEPATH "Path to nvcc")

include(GNUInstallDirs)

find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(GSL REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(driveless REQUIRED)


if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
    message(STATUS "CUDA Toolkit version is >= 13.0 (found ${CUDAToolkit_VERSION})")
endif()

string(REPLACE "." ";" CUDA_VER_LIST ${CMAKE_CUDA_COMPILER_VERSION})
list(GET CUDA_VER_LIST 0 CUDA_MAJOR)
list(GET CUDA_VER_LIST 1 CUDA_MINOR)

add_compile_definitions(CUDA_VERSION_MAJOR=${CUDA_MAJOR})
add_compile_definitions(CUDA_VERSION_MINOR=${CUDA_MINOR})


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
link_directories(/usr/local/lib/driveless)
# Library sources

add_library(
  ${CMAKE_PROJECT_NAME} SHARED
  src/cuda/kinematics.cu
  src/cuda/path_feasible.cu
  src/cuda/random_number_gen.cu
  src/cuda/graph.cu
  src/cuda/graph_list.cu
  src/cuda/graph_best_node.cu
  src/cuda/graph_optimization.cu
  src/cuda/graph_expansion.cu
  src/cuda/graph_nodes_in_range.cu
  src/cuda/graph_costs.cu
  src/cuda/apf.cu
  src/cuda/graph_expansion_search_zones.cu
  src/cuda/graph_collision.cu
  src/cuda/graph_hermite_connection.cu
  src/cuda/graph_direct_goal_connection.cu
  src/cuda/graph_check_goal_reached.cu
  src/fastrrt.cpp
  src/python_bind.cpp
  src/python_cuda_graph_bind.cpp
  src/interpolation.cpp
  src/graph_check_dag_consistency.cpp
  #src/debug.cpp
)

# Properties for CUDA shared lib
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    CUDA_RUNTIME_LIBRARY Shared
)

# Include paths
target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${CUDAToolkit_INCLUDE_DIRS}
)

# Link dependencies
if(CUDA_LIB_JETSON)
    target_link_libraries(${PROJECT_NAME}
        PRIVATE
            Threads::Threads
            GSL::gsl
            GSL::gslcblas
            jetson-utils
            driveless
            #CUDA::cudart
    )
else()
    target_link_libraries(${PROJECT_NAME}
        PRIVATE
            Threads::Threads
            GSL::gsl
            GSL::gslcblas
            driveless
            #CUDA::cudart
    )
endif()

# Install library
install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/driveless
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/driveless
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/driveless
)

# Install headers
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/driveless
)

# Export set for find_package()
install(EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

# Generate package config files
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)




