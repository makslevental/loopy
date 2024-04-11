#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
// #include "Transform/CommonExtensions/CommonExtensions.h"
// #include "Transform/Structured/StructuredTransformOpsExt.h"
// #include "Transform/LLVMGPU/LLVMGPUExtensions.h"

namespace py = pybind11;
using namespace llvm;
using namespace mlir;
using namespace mlir::python::adaptors;

namespace py = pybind11;
using namespace mlir;

PYBIND11_MODULE(_site_initialize_0, m) {
  m.def("register_openmp_dialect_translation",
        [](MlirDialectRegistry &registry) {
          auto rgtry = unwrap(registry);
          mlir::registerOpenMPDialectTranslation(*rgtry);
        });
  m.def("register_dialects", [](MlirDialectRegistry &registry) {
    // auto rgtry = unwrap(registry);
    // mlirRegisterAllLLVMTranslations
    // mlir::registerOpenMPDialectTranslationnslation(*rgtry);
  });
  m.def("context_init_hook", [](MlirContext &ctx) {
    auto context = unwrap(ctx);
    mlir::registerOpenMPDialectTranslation(*context);
    //    mlir::registerTransformDialectCommonExtension(*context);
    //    mlir::registerTransformDialectStructuredExtension(*context);
    //    mlir::registerTransformDialectLLVMGPUExtension(*context);
  });
}
