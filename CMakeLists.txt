cmake_minimum_required(VERSION 2.6)

project(cudaSift)
set(cudaSift_VERSION_MAJOR 2)
set(cudaSift_VERSION_MINOR 0)
set(cudaSift_VERSION_PATCH 0)

set(CPACK_PACKAGE_VERSION_MAJOR "${cudaSift_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${cudaSift_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${cudaSift_VERSION_PATCH}")
set(CPACK_GENERATOR "ZIP")
include(CPack)

option(CUDASIFT_WITH_OPENCV "CUDASIFT_WITH_OPENCV" ON)
option(CUDASIFT_WITH_LIB "CUDASIFT_WITH_LIB" ON)

find_package(OpenCV)
if (NOT OpenCV)
    set(CUDASIFT_WITH_OPENCV OFF)
endif()
find_package(CUDA)
if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

if (WIN32)
  set(EXTRA_CXX_FLAGS "/DVERBOSE /D_CRT_SECURE_NO_WARNINGS ")
  list(APPEND CUDA_NVCC_FLAGS "--compiler-options;-O2;-DVERBOSE") 
endif()
if (UNIX)
  if (APPLE)
    set(EXTRA_CXX_FLAGS "-DVERBOSE -msse2")
    list(APPEND CUDA_NVCC_FLAGS "--compiler-options;-O2;-DVERBOSE") 
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -msse2 -fPIC")
    list(APPEND CUDA_NVCC_FLAGS "-lineinfo;--compiler-options;-O2;-DVERBOSE") 
  endif()
endif()

set(cuda_sources
  # dynamic.cu
  cudaImage.cu  
  cudaImage.h  
  cudaSiftH.cu 
  cudaSiftH.h  
  matching.cu  
  cudaSiftD.h  
  cudaSift.h  
  cudautils.h
)
set(cuda_sources_absolute "")
foreach(f ${cuda_sources})
  get_filename_component(a ${f} ABSOLUTE)
  list(APPEND cuda_sources_absolute ${a})
endforeach(f)

set(sources
  geomFuncs.cpp  
  mainSift.cpp
)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR} 
)

SET(CUDA_SEPARABLE_COMPILATION ON)

if (CUDASIFT_WITH_OPENCV)
  cuda_add_executable(cudasift ${cuda_sources} ${sources})
  set_target_properties(cudasift PROPERTIES
    COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"			   
  )

  target_link_libraries(cudasift
    /usr/local/cuda/lib64/libcudadevrt.a ${OpenCV_LIBS} 
  )
 
endif()

add_subdirectory(lib)

install(FILES 
  ${cuda_sources} 
  ${sources}
  cudaSiftD.cu
  CMakeLists.txt
  DESTINATION .
)
install(FILES data/left.pgm data/righ.pgm
  DESTINATION data
)
