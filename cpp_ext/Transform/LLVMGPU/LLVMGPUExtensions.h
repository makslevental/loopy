// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TRANSFORMEXTENSIONS_LLVMGPUEXTENSIONS_H_
#define TRANSFORMEXTENSIONS_LLVMGPUEXTENSIONS_H_

#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"

namespace mlir {
class DialectRegistry;

namespace func {
class FuncOp;
}

namespace scf {
class ForallOp;
class IfOp;
class ForOp;
} // namespace scf

namespace vector {
class VectorDialect;
class WarpExecuteOnLane0Op;
} // namespace vector

/// Registers Flow transformations that require IREE-specific information into
/// the transform dialect.
void registerTransformDialectLLVMGPUExtension(DialectRegistry &registry);
void registerTransformDialectLLVMGPUExtension(MLIRContext &context);

namespace transform {
// Hook to register LLVMGPU transformations to the transform dialect.
class LLVMGPUExtensions
    : public transform::TransformDialectExtension<LLVMGPUExtensions> {
public:
  LLVMGPUExtensions();
};
} // namespace transform

/// Transformation to convert scf.forall to gpu distribution.
FailureOr<SmallVector<OpFoldResult>>
rewriteForallToGpu(scf::ForallOp forallOp,
                   const SmallVector<int64_t> &globalWorkgroupSizes,
                   RewriterBase &rewriter, bool syncAfterDistribute = true);

class SimplePatternRewriter : public PatternRewriter {
public:
  SimplePatternRewriter(MLIRContext *context) : PatternRewriter(context) {}

  SimplePatternRewriter(Operation *op) : PatternRewriter(op->getContext()) {
    setInsertionPoint(op);
  }
};

} // namespace mlir

#define GET_OP_CLASSES
#include "LLVMGPUExtensionsOps.h.inc"

#endif // TRANSFORMEXTENSIONS_LLVMGPUEXTENSIONS_H_
