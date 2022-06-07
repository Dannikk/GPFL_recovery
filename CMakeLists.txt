cmake_minimum_required(VERSION 3.20)
project(GPFL_recovery)

set(CMAKE_CXX_STANDARD 17)

add_definitions(-DNO_DLL)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c99 -fopenmp")

find_package(OpenMP)
find_package( OpenCV REQUIRED )
find_package(Boost 1.40 COMPONENTS program_options REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(GPFL_recovery main.cpp cgp.c Image4CGP.cpp EdgeDetection.cpp EdgeDetection.h)

#target_link_libraries( GPFL_recovery ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

target_link_libraries( GPFL_recovery ${OpenCV_LIBS} OpenMP::OpenMP_CXX Boost::program_options)

message(STATUS "Checking OpenMP")
find_package(OpenMP)
IF(OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    #    message("Found OpenMP!")
endif()