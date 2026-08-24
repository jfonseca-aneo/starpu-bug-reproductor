cmake_minimum_required(VERSION 3.16)
project(starpu_halo_ring_repro C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(PkgConfig REQUIRED)
# StarPU's .pc files are versioned (e.g. starpumpi-1.4); bump STARPU_PC_VERSION
# if your distro ships a different StarPU series.
set(STARPU_PC_VERSION "1.4" CACHE STRING "StarPU pkg-config version suffix")
pkg_check_modules(STARPU REQUIRED IMPORTED_TARGET starpumpi-${STARPU_PC_VERSION})
find_package(MPI REQUIRED)

add_executable(starpu_halo_ring main.c)
target_link_libraries(starpu_halo_ring PkgConfig::STARPU MPI::MPI_C)

add_executable(starpu_halo_ring_detached main_detached.c)
target_link_libraries(starpu_halo_ring_detached PkgConfig::STARPU MPI::MPI_C)
