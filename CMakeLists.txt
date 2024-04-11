cmake_minimum_required(VERSION 3.13.4)

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 NEW)
endif()

project(nelli LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD
    17
    CACHE STRING "C++ standard to conform to")

set(LLVM_INSTALL_DIR
    ""
    CACHE STRING "Path to LLVM install dir")
if(LLVM_INSTALL_DIR STREQUAL "")
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import mlir; print(mlir.__path__[0])"
    OUTPUT_VARIABLE LLVM_INSTALL_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
include(CMakePrintHelpers)
cmake_print_variables(LLVM_INSTALL_DIR)
list(APPEND CMAKE_PREFIX_PATH ${LLVM_INSTALL_DIR})
cmake_print_variables(CMAKE_PREFIX_PATH)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

include(MLIRDetectPythonEnv)
set(MLIR_INCLUDE_TESTS 0)

add_subdirectory(cpp_ext)
