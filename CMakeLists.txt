cmake_minimum_required(VERSION 3.18)
project(equilibrium_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_native
  src/python_bindings.cpp
  src/solver_core.cpp
)

target_include_directories(_native PRIVATE include)
install(TARGETS _native DESTINATION equilibrium_cpp)
