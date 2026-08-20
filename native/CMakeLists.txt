cmake_minimum_required(VERSION 3.28)
project(ltx_cuda VERSION 1.0.0 LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES must name a real SASS target (for example 120-real)")
endif()

add_library(ltx_cuda SHARED src/ltx_cuda.cu)
target_include_directories(ltx_cuda PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(ltx_cuda PRIVATE cxx_std_17 cuda_std_17)
set_target_properties(ltx_cuda PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  CUDA_VISIBILITY_PRESET hidden
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME ltx_cuda
)

install(TARGETS ltx_cuda LIBRARY DESTINATION lib)
install(FILES include/ltx_cuda.h DESTINATION include)
