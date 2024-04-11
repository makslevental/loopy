include(AddMLIRPython)
mlir_configure_python_dev_packages()
mlir_detect_pybind11_install()

set(NELLI_MLIR_PYTHON_PACKAGES_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/python_packages")

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=nelli.mlir._mlir.")
declare_mlir_python_sources(NelliPythonSources)

include_directories(.)

add_subdirectory(TilingInterface)
add_subdirectory(RefBackend)
add_subdirectory(Transform)
add_subdirectory(LinalgTransforms)
add_subdirectory(FakeQuantize)
add_subdirectory(GPUTransforms)
add_subdirectory(PDLByteCodeInterpreter)
add_subdirectory(LLVMGPUVectorLowering)

declare_mlir_python_extension(
  NelliPythonSources.Extension
  MODULE_NAME
  _nelli_mlir
  ADD_TO_PARENT
  NelliPythonSources
  SOURCES
  NelliExtension.cpp
  LoopUtils.cpp
  AffineAnalysis.cpp
  PRIVATE_LINK_LIBS
  MLIRAnalysis
  MLIRPresburger
  MLIRAffineAnalysis
  MLIRSparseTensorPipelines
  TilingInterfacePasses
  RefBackend
  TransformDialectInterpreter
  LinalgTransforms
  FakeQuantize
  GPUTransforms
  PDLByteCodeInterpreter
  LLVMGPUVectorLowering)

# add_subdirectory(openmp_translation)

declare_mlir_python_extension(
  NelliPythonSources.SiteInitialize
  MODULE_NAME
  _site_initialize_0
  ADD_TO_PARENT
  NelliPythonSources
  SOURCES
  SiteInitialize.cpp
  PRIVATE_LINK_LIBS
  LLVMSupport
  MLIRIR)
target_link_libraries(
  NelliPythonSources.SiteInitialize
  INTERFACE
  MLIROpenMPToLLVMIRTranslation
#  CommonExtensions
#  StructuredTransformOpsExt
#  LLVMGPUExtensions
)

set(PYBINDINGS_SRC "${MLIR_INSTALL_PREFIX}/src/python/MLIRPythonExtension.Core")
target_include_directories(NelliPythonSources.Extension
                           INTERFACE "${PYBINDINGS_SRC}")

execute_process(
  COMMAND python ${CMAKE_SOURCE_DIR}/scripts/regex_mlir_bindings.py
          ${PYBINDINGS_SRC}/IRModule.h)
execute_process(
  COMMAND python ${CMAKE_SOURCE_DIR}/scripts/regex_mlir_bindings.py
          ${PYBINDINGS_SRC}/IRCore.cpp)
execute_process(
  COMMAND python ${CMAKE_SOURCE_DIR}/scripts/regex_mlir_bindings.py
          ${PYBINDINGS_SRC}/IRAffine.cpp)

set(_source_components
    MLIRPythonSources MLIRPythonExtension.Core
    MLIRPythonExtension.RegisterEverything NelliPythonSources)

add_mlir_python_common_capi_library(
  NelliMLIRAggregateCAPI
  INSTALL_COMPONENT
  NelliMLIRPythonModules
  INSTALL_DESTINATION
  _mlir/_mlir_libs
  OUTPUT_DIRECTORY
  "${NELLI_MLIR_PYTHON_PACKAGES_DIR}/nelli/mlir/_mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT
  "../../../../.."
  DECLARED_SOURCES
  ${_source_components})

add_mlir_python_modules(
  NelliPythonModules
  ROOT_PREFIX
  "${NELLI_MLIR_PYTHON_PACKAGES_DIR}/nelli/mlir/_mlir"
  INSTALL_PREFIX
  "_mlir"
  DECLARED_SOURCES
  ${_source_components}
  COMMON_CAPI_LINK_LIBS
  NelliMLIRAggregateCAPI)
