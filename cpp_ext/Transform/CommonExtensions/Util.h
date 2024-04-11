//
// Created by maksim on 3/21/23.
//

#ifndef NELLI_UTIL_H
#define NELLI_UTIL_H

#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {

/// Creates an allocation in the entry block of the function if the size is
/// statically bounded. For a static allocation, it returns an allocation
/// of the same size but in the entry basic block. For dynamic (still bounded)
/// allocations creates an allocation, and inserts a subview to match the
/// dynamic shape of the allocation. Returns std::nullopt if the method
/// couldnt creat an allocation in the entry block.
template <typename AllocLikeOpType>
std::optional<Value>
hoistOneStaticallyBoundAllocation(func::FuncOp funcOp, OpBuilder &builder,
                                  Location loc, MemRefType allocaType,
                                  ValueRange dynamicSizes,
                                  std::optional<uint64_t> alignment);

/// Hoists `allocaOp` to the entry block of the function if the size is
/// statically bounded. For a static allocation, it returns an allocation
/// of the same size but in the entry basic block. For dynamic (still bounded)
/// allocations creates an allocation, and inserts a subview to match the
/// dynamic shape of the allocation. The method returns a value, but
/// does not replace the uses of the `allocaOp`.
template <typename AllocLikeOpType>
std::optional<Value>
hoistOneStaticallyBoundAllocation(func::FuncOp funcOp, OpBuilder &builder,
                                  AllocLikeOpType allocaOp);

/// Traverse funcOp and try to hoist every AllocaOp to the entry block of the
/// function if the size is statically bounded.
template <typename AllocLikeOpType>
void hoistStaticallyBoundAllocationsInFunc(RewriterBase &rewriter,
                                           func::FuncOp funcOp);

Operation *createLinalgCopyOp(OpBuilder &b, Location loc, Value from, Value to,
                              ArrayRef<NamedAttribute> attributes = {});

std::optional<SmallVector<int64_t>> getWmmaNativeVectorSize(Operation *op);
std::optional<SmallVector<int64_t>>
gpuMmaUnrollOrder(vector::ContractionOp contract);
std::optional<SmallVector<int64_t>> getMmaNativeVectorSize(Operation *op);

void addTransformPostBufferizationPasses(OpPassManager &passManager);
} // namespace mlir

namespace nelli {
mlir::LogicalResult eliminateEmptyTensors(
    mlir::Operation *op,
    const mlir::bufferization::OneShotBufferizationOptions &options);
}

#endif // NELLI_UTIL_H
