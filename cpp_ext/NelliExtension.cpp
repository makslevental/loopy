#include "IRModule.h"
#include "mlir-c/AffineExpr.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/IR.h"
#include "mlir/CAPI/AffineExpr.h"
#include "mlir/CAPI/AffineMap.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/Operation.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include <mlir/Dialect/Affine/LoopUtils.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include "AffineAnalysis.h"
#include "FakeQuantize/FakeQuantize.h"
#include "LinalgTransforms/LinalgTransforms.h"
#include "LoopUtils.h"
#include "Pybind.h"
#include "RaiseToAffine/RaiseToAffine.h"
#include "RefBackend/RefBackend.h"
#include "TilingInterface/TilingInterface.h"
#include "Transform/TransformDialectInterpreter.h"
#include "utils.h"
#include <string>

namespace py = pybind11;
using namespace mlir::python;
using namespace mlir;
using namespace presburger;
using namespace tabulate;

static mlir::LogicalResult
getOpIndexSet(mlir::Operation *op, mlir::FlatAffineValueConstraints *indexSet) {
  llvm::SmallVector<mlir::Operation *, 4> ops;
  mlir::getEnclosingAffineOps(*op, &ops);
  return getIndexSet(ops, indexSet);
}

class PyAffineMapAttribute : public PyConcreteAttribute<PyAffineMapAttribute> {
public:
  using PyConcreteAttribute::PyConcreteAttribute;
};

template <typename T> T *unwrapApiObject(const py::handle apiObject) {
  return unwrap(mlirPythonCapsuleToOperation(
      py::detail::mlirApiObjectToCapsule(apiObject).ptr()));
}

template <typename T> T unwrapOpObject(const py::handle apiObject) {
  auto *op = unwrapApiObject<mlir::Operation>(apiObject);
  return llvm::dyn_cast<T>(op);
}

py::object getOpView(MlirOperation op) {
  auto ctx = PyMlirContext::forContext(mlirOperationGetContext(op));
  auto pyFoundOp = PyOperation::forOperation(ctx, op);
  return pyFoundOp->createOpView();
}

py::dict getBoundsFromRelation(const mlir::FlatAffineRelation &relation) {
  py::dict bounds;
  auto nds = relation.getNumDimAndSymbolVars();
  for (unsigned i = 0; i < nds; ++i) {
    py::dict bound;
    if (relation.hasValue(i)) {
      auto LB =
          relation.getConstantBound(mlir::presburger::IntegerRelation::LB, i);
      auto UB =
          relation.getConstantBound(mlir::presburger::IntegerRelation::UB, i);
      auto EQ =
          relation.getConstantBound(mlir::presburger::IntegerRelation::EQ, i);
      if (LB.has_value()) {
        bound["LB"] = int64FromMPInt(LB.value());
      } else {
        bound["LB"] = py::none();
      }
      if (UB.has_value()) {
        bound["UB"] = int64FromMPInt(UB.value());
      } else {
        bound["UB"] = py::none();
      }
      if (EQ.has_value()) {
        bound["EQ"] = int64FromMPInt(EQ.value());
      } else {
        bound["EQ"] = py::none();
      }
      bounds[py::cast<>(wrap(relation.getValue(i)))] = bound;
    }
  }
  return bounds;
}

thread_local py::object annotator_;

PYBIND11_MODULE(_nelli_mlir, m) {
  auto mod = py::module_::import(MAKE_MLIR_PYTHON_QUALNAME("ir"));
  PyArithValue::bind(m);
  PyMemRefValue::bind(m);
  PyTensorValue::bind(m);

  m.def("walk_affine_exprs",
        [](PyAffineMap &self,
           std::function<void(size_t resIdx, MlirAffineExpr expr)> callback) {
          for (const auto &idx_expr :
               llvm::enumerate(unwrap(self.get()).getResults())) {
            auto idx = idx_expr.index();
            auto expr = idx_expr.value();
            expr.walk([&callback, &idx](mlir::AffineExpr expr) {
              callback(idx, wrap(expr));
            });
          }
        });

  m.def("walk_operation", [](PyOperation &self,
                             std::function<void(MlirOperation)> callback) {
    unwrap(self.get())->walk<WalkOrder::PreOrder>([&callback](Operation *op) {
      callback(wrap(op));
    });
  });

  m.def("get_affine_map_from_attr", [](PyAttribute &self) {
    auto aff_map =
        PyAffineMap(self.getContext(), mlirAffineMapAttrGetValue(self.get()));
    return aff_map.get();
  });

  m.def("show_value_as_operand", [](const py::handle valueApiObject) {
    auto capsule = pybind11::detail::mlirApiObjectToCapsule(valueApiObject);
    MlirValue mlirValue = mlirPythonCapsuleToValue(capsule.ptr());
    return nelli::showValueAsOperand(unwrap(mlirValue));
  });
  m.def("get_affine_value_map", [](const py::handle affineOpApiObject) {
    auto affineApplyOp = unwrapOpObject<mlir::AffineApplyOp>(affineOpApiObject);
    mlir::AffineValueMap valueMap;
    valueMap = affineApplyOp.getAffineValueMap();
    py::list dims;
    py::list syms;
    for (unsigned int i = 0; i < valueMap.getNumDims(); ++i) {
      auto v = valueMap.getOperand(i);
      dims.append(wrap(v));
    }
    for (unsigned int i = valueMap.getNumDims();
         i < valueMap.getNumDims() + valueMap.getNumSymbols(); ++i) {
      auto v = valueMap.getOperand(i);
      syms.append(wrap(v));
    }
    return py::make_tuple(dims, syms);
  });
  m.def("get_access_relation", [](const py::handle affineOpApiObject) {
    auto *op = unwrapApiObject<mlir::Operation>(affineOpApiObject);
    mlir::MemRefAccess *access;
    access = new mlir::MemRefAccess(op);
    py::dict indices;
    for (const auto &pos_idx : llvm::enumerate(access->indices)) {
      indices[py::cast<>(pos_idx.index())] = py::cast<>(wrap(pos_idx.value()));
    }
    mlir::FlatAffineValueConstraints domain;
    (void)getOpIndexSet(op, &domain);
    mlir::FlatAffineRelation domainRel(domain.getNumDimVars(),
                                       /*numRangeDims=*/0, domain);
    auto bounds = getBoundsFromRelation(domainRel);
    return py::make_tuple(bounds, indices);
  });

  m.def("show_access_relation",
        [](const py::handle srcOpApiObject, const py::handle dstOpApiObject) {
          auto *srcOp = unwrapApiObject<mlir::Operation>(srcOpApiObject);
          auto *dstOp = unwrapApiObject<mlir::Operation>(dstOpApiObject);
          nelli::myCheckDependenceSrcDst(srcOp, dstOp);
        });

  m.def("show_sanity_check_access_relation",
        [](const py::handle srcOpApiObject, const py::handle dstOpApiObject) {
          auto *srcOp = unwrapApiObject<mlir::Operation>(srcOpApiObject);
          auto *dstOp = unwrapApiObject<mlir::Operation>(dstOpApiObject);
          nelli::sanityCheckDependenceSrcDst(srcOp, dstOp);
        });
  m.def("reset_disambig_names", []() { nelli::seen.clear(); });

  m.def("get_common_loops",
        [](const py::handle srcOpApiObject, const py::handle dstOpApiObject)
            -> std::optional<std::vector<py::object>> {
          auto *srcOp = unwrapApiObject<mlir::Operation>(srcOpApiObject);
          auto *dstOp = unwrapApiObject<mlir::Operation>(dstOpApiObject);
          MemRefAccess srcAccess(srcOp);
          MemRefAccess dstAccess(dstOp);
          FlatAffineRelation srcRel, dstRel;
          if (failed(srcAccess.getAccessRelation(srcRel)))
            return {};
          if (failed(dstAccess.getAccessRelation(dstRel)))
            return {};
          FlatAffineValueConstraints srcDomain = srcRel.getDomainSet();
          FlatAffineValueConstraints dstDomain = dstRel.getDomainSet();
          std::vector<py::object> resVec{};
          for (const AffineForOp &forOp :
               nelli::getCommonLoops(srcDomain, dstDomain)) {
            auto mlirForOp = wrap(forOp);
            resVec.emplace_back(getOpView(mlirForOp));
          }
          return {resVec};
        });

  m.def("get_loop_bounds", [](const py::handle srcOpApiObject) {
    auto affForOp = unwrapOpObject<mlir::AffineForOp>(srcOpApiObject);

    mlir::FlatAffineRelation lowerRel;
    auto lowerMap = affForOp.getLowerBoundMap();
    (void)getRelationFromMap(lowerMap, lowerRel);
    mlir::FlatAffineRelation upperRel;
    auto upperMap = affForOp.getUpperBoundMap();
    (void)getRelationFromMap(upperMap, upperRel);
    py::dict bounds;
    py::dict bound;
    auto LB =
        lowerRel.getConstantBound(mlir::presburger::IntegerRelation::LB, 0);
    auto UB =
        upperRel.getConstantBound(mlir::presburger::IntegerRelation::UB, 0);
    if (LB.has_value()) {
      bound["LB"] = int64FromMPInt(LB.value());
    } else {
      bound["LB"] = py::none();
    }
    if (UB.has_value()) {
      bound["UB"] = int64FromMPInt(UB.value());
    } else {
      bound["UB"] = py::none();
    }
    bound["EQ"] = py::none();
    bounds[py::cast<>(wrap(affForOp.getInductionVar()))] = bound;

    return bounds;
  });

  m.def("get_opview", [](const MlirOperation op) { return getOpView(op); });

  m.def("show_direction_vector", [](const py::handle srcOpApiObject,
                                    const py::handle dstOpApiObject,
                                    int toLoopDepth) {
    auto *srcOp = unwrapApiObject<mlir::Operation>(srcOpApiObject);
    auto *dstOp = unwrapApiObject<mlir::Operation>(dstOpApiObject);
    unsigned numCommonLoops = getNumCommonSurroundingLoops(*srcOp, *dstOp);
    MemRefAccess srcAccess(srcOp);
    MemRefAccess dstAccess(dstOp);
    FlatAffineValueConstraints dependenceConstraints;
    SmallVector<DependenceComponent, 2> dependenceComponents;
    DependenceResult result = checkMemrefAccessDependence(
        srcAccess, dstAccess, toLoopDepth, &dependenceConstraints,
        &dependenceComponents, true);
    bool ret = hasDependence(result);
    return nelli::getDirectionVectorStr(ret, numCommonLoops, toLoopDepth,
                                        dependenceComponents);
  });

  m.def("affine_for_skew", [](const py::handle forOpApiObject,
                              const std::vector<uint64_t> &shifts) {
    auto forOp = unwrapOpObject<mlir::AffineForOp>(forOpApiObject);
    if (failed(mlir::affineForOpBodySkew(forOp, shifts))) {
      throw py::value_error("skew failed");
    }
  });

  m.def("affine_for_unroll_by_factor", [](const py::handle forOpApiObject,
                                          int unrollFactor,
                                          const py::object &annotator) {
    auto forOp = unwrapOpObject<mlir::AffineForOp>(forOpApiObject);
    llvm::function_ref<void(unsigned, Operation *, OpBuilder)> annotateFn =
        nullptr;
    using Annotation = std::pair<std::string, MlirAttribute>;
    annotator_ = py::reinterpret_borrow<py::object>(annotator);
    if (!annotator_.is(py::none())) {
      annotateFn = [](unsigned i, Operation *op, OpBuilder b) {
        auto res = annotator_(i, wrap(op));
        if (!res.is(py::none())) {
          auto annot = py::cast<Annotation>(res);
          op->setAttr(annot.first, unwrap(annot.second));
        }
      };
    }
    if (failed(mlir::loopUnrollByFactor(forOp, unrollFactor, annotateFn))) {
      throw py::value_error("unroll by factor failed");
    }
  });
  m.def("print_help", []() -> std::string {
    PassPipelineCLParser passPipeline("", "Compiler passes to run", "p");
    std::string dummy = "dummy";
    std::string help = "--help";
    char *argv[] = {dummy.data(), help.data()};
    llvm::cl::ParseCommandLineOptions(2, argv, "");
  });

  nelli::registerTilingInterfacePass();
  nelli::registerMungeCallingConventionPass();
  nelli::registerMungeMemrefCopyPass();
  nelli::registerGeneralizeTensorPadPass();
  nelli::registerTransformDialectInterpreterPass();
  nelli::registerTransformDialectEraseSchedulePass();
  nelli::registerRaiseSCFToAffinePass();
  nelli::registerLinalgTransforms();
  nelli::registerLinalgFakeQuantizePass();
}
