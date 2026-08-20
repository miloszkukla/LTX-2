cmake_minimum_required(VERSION 3.25)
project(ltx_oiio LANGUAGES CXX)

find_package(OpenImageIO REQUIRED)
add_library(ltx_oiio SHARED ltx_oiio.cpp)
target_compile_features(ltx_oiio PRIVATE cxx_std_17)
target_link_libraries(ltx_oiio PRIVATE OpenImageIO::OpenImageIO)
set_target_properties(ltx_oiio PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
