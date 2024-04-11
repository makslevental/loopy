#add_subdirectory(CommonExtensions)
#add_subdirectory(Structured)
#add_subdirectory(LLVMGPU)

add_mlir_library(
  DialectsTransforms
  ListenerCSE.cpp
  TransformMatchers.cpp
  PARTIAL_SOURCES_INTENDED
  LINK_LIBS
  PRIVATE
  # TODO: break dialect dependency by implementing the transformation separately
  # and registering it.
  MLIRAsyncDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMathDialect
  MLIRIR
  MLIRPass
  MLIRPDLDialect
  MLIRTransformDialect
  MLIRTransformDialectTransforms)

add_mlir_library(
  TransformDialectInterpreter
  TransformDialectInterpreter.cpp
  EXCLUDE_FROM_LIBMLIR
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRPass
  MLIRPDLDialect
  MLIRTransformDialect
  MLIRTransformDialectTransforms)
