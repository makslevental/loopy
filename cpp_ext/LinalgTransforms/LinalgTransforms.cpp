//===- LinalgTransforms.cpp - Test Linalg transformation patterns -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for ing Linalg transformations.
//
//===----------------------------------------------------------------------===//

#include "LinalgTransforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::affine;

namespace {
struct LinalgTransforms
    : public PassWrapper<LinalgTransforms, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgTransforms)

  LinalgTransforms() = default;
  LinalgTransforms(const LinalgTransforms &pass) : PassWrapper(pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<AffineDialect,
                    bufferization::BufferizationDialect,
                    memref::MemRefDialect,
                    scf::SCFDialect,
                    linalg::LinalgDialect,
                    vector::VectorDialect,
                    gpu::GPUDialect>();
    // clang-format on
  }
  StringRef getArgument() const final { return "linalg-transform-patterns"; }
  StringRef getDescription() const final {
    return "Linalg transformation patterns by applying them greedily.";
  }

  void runOnOperation() override;

  Option<bool> patterns{*this, "patterns",
                        llvm::cl::desc("a mixed set of patterns"),
                        llvm::cl::init(false)};
  Option<bool> vectorTransferForwardingPatterns{
      *this, "vector-transfer-forwarding-patterns",
      llvm::cl::desc(
          "a fused pass that forwards memref.copy to vector.transfer"),
      llvm::cl::init(false)};
  Option<bool> genericToVectorPattern{
      *this, "linalg-to-vector-patterns",
      llvm::cl::desc("a set of patterns that rewrite a linalg contraction "
                     "in vector.contract form"),
      llvm::cl::init(false)};
  Option<bool> transformPadTensor{
      *this, "transform-pad-tensor",
      llvm::cl::desc("transform pad tensor by copying with generic ops"),
      llvm::cl::init(false)};
  Option<bool> generalizePadTensor{
      *this, "generalize-pad-tensor",
      llvm::cl::desc("transform pad tensor by copying with generic ops"),
      llvm::cl::init(false)};
  Option<bool> generalizeTensorPackOp{
      *this, "generalize-tensor-pack",
      llvm::cl::desc("transform that generalizes pack ops into a sequence "
                     "of tensor and Linalg ops"),
      llvm::cl::init(false)};
  Option<bool> generalizeTensorUnPackOp{
      *this, "generalize-tensor-unpack",
      llvm::cl::desc("transform that generalizes unpack ops into a sequence "
                     "of tensor and Linalg ops"),
      llvm::cl::init(false)};
  Option<bool> swapSubTensorPadTensor{
      *this, "swap-subtensor-padtensor",
      llvm::cl::desc("rewrite of subtensor(tensor.pad) into "
                     "tensor.pad(subtensor)"),
      llvm::cl::init(false)};
  ListOption<int64_t> peeledLoops{
      *this, "peeled-loops",
      llvm::cl::desc("Loops to be peeled when tile-pattern")};
  ListOption<int64_t> tileSizes{
      *this, "tile-sizes",
      llvm::cl::desc("Linalg tile sizes for tile-pattern")};
  Option<bool> skipPartial{
      *this, "skip-partial",
      llvm::cl::desc("Skip loops inside partial iterations during peeling"),
      llvm::cl::init(false)};
  Option<std::string> loopType{
      *this, "loop-type",
      llvm::cl::desc("Specify the type of loops to generate: for, parallel or "
                     "tiled_loop"),
      llvm::cl::init("for")};
  Option<bool> bubbleUpExtractSliceOpPattern{
      *this, "bubble-up-extract-slice-op-pattern",
      llvm::cl::desc("rewrite of linalgOp + extract_slice into "
                     "extract_slice + linalgOp"),
      llvm::cl::init(false)};
  Option<bool> swapExtractSliceWithFill{
      *this, "swap-extract-slice-with-fill-pattern",
      llvm::cl::desc("patterns to swap tensor.extract_slice(linalg.fill())"),
      llvm::cl::init(false)};
  Option<bool> eraseUnusedOperandsAndResults{
      *this, "erase-unused-operands-and-results",
      llvm::cl::desc("patterns to erase unused operands and results"),
      llvm::cl::init(false)};
  Option<bool> eraseUnnecessaryInputs{
      *this, "erase-unnecessary-inputs",
      llvm::cl::desc("patterns to erase unnecessary inputs"),
      llvm::cl::init(false)};
  Option<bool> decomposeConvolutions{
      *this, "decompose-convolutions",
      llvm::cl::desc("patterns to decompose convolutions"),
      llvm::cl::init(false)};
  Option<bool> conv2DToImg2Col{
      *this, "conv2d-im2col",
      llvm::cl::desc(
          "patterns to perform im2col transformations on convolutions"),
      llvm::cl::init(false)};
};
} // namespace

static void applyPatterns(func::FuncOp funcOp) {
  MLIRContext *ctx = funcOp.getContext();
  RewritePatternSet patterns(ctx);

  //===--------------------------------------------------------------------===//
  // Linalg distribution patterns.
  //===--------------------------------------------------------------------===//
  LinalgLoopDistributionOptions distributionOptions;

  //===--------------------------------------------------------------------===//
  // Linalg to vector contraction patterns.
  //===--------------------------------------------------------------------===//
  patterns.add<CopyVectorizationPattern>(ctx);

  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

static void applyVectorTransferForwardingPatterns(func::FuncOp funcOp) {
  RewritePatternSet forwardPattern(funcOp.getContext());
  forwardPattern.add<LinalgCopyVTRForwardingPattern>(funcOp.getContext());
  forwardPattern.add<LinalgCopyVTWForwardingPattern>(funcOp.getContext());
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(forwardPattern));
}

static void applyLinalgToVectorPatterns(func::FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  auto *ctx = funcOp.getContext();
  patterns.add<CopyVectorizationPattern>(ctx);
  populatePadOpVectorizationPatterns(patterns);
  populateConvolutionVectorizationPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

static void applyPadTensorToGenericPatterns(func::FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
//  patterns.add<PadOpTransformationPattern>(funcOp.getContext());
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

static void applyGeneralizePadTensorPatterns(func::FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  patterns.add<GeneralizePadOpPattern>(funcOp.getContext());
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

static void applyGeneralizeTensorPackPatterns(func::FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  patterns.add<GeneralizeOuterUnitDimsPackOpPattern>(funcOp.getContext());
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

static void applyGeneralizeTensorUnPackPatterns(func::FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  patterns.add<GeneralizeOuterUnitDimsUnPackOpPattern>(funcOp.getContext());
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

static void applyExtractSliceOfPadTensorSwapPattern(func::FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  patterns.add<ExtractSliceOfPadTensorSwapPattern>(funcOp.getContext());
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

static void applyBubbleUpExtractSliceOpPattern(func::FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  populateBubbleUpExtractSliceOpPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

static void applySwapExtractSliceWithFillPattern(func::FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  populateSwapExtractSliceWithFillPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

static void applyEraseUnusedOperandsAndResultsPatterns(func::FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  populateEraseUnusedOperandsAndResultsPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

static void applyEraseUnnecessaryInputs(func::FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  populateEraseUnnecessaryInputsPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

static void applyDecomposeConvolutions(func::FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  populateDecomposeConvolutionPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

static void applyConv2DToImg2Col(func::FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  populateConvertConv2DToImg2ColPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

/// Apply transformations specified as patterns.
void LinalgTransforms::runOnOperation() {
  if (patterns)
    return applyPatterns(getOperation());
  if (vectorTransferForwardingPatterns)
    return applyVectorTransferForwardingPatterns(getOperation());
  if (genericToVectorPattern)
    return applyLinalgToVectorPatterns(getOperation());
  if (transformPadTensor)
    return applyPadTensorToGenericPatterns(getOperation());
  if (generalizePadTensor)
    return applyGeneralizePadTensorPatterns(getOperation());
  if (generalizeTensorPackOp)
    return applyGeneralizeTensorPackPatterns(getOperation());
  if (generalizeTensorUnPackOp)
    return applyGeneralizeTensorUnPackPatterns(getOperation());
  if (swapSubTensorPadTensor)
    return applyExtractSliceOfPadTensorSwapPattern(getOperation());
  if (bubbleUpExtractSliceOpPattern)
    return applyBubbleUpExtractSliceOpPattern(getOperation());
  if (swapExtractSliceWithFill)
    return applySwapExtractSliceWithFillPattern(getOperation());
  if (eraseUnusedOperandsAndResults)
    return applyEraseUnusedOperandsAndResultsPatterns(getOperation());
  if (eraseUnnecessaryInputs)
    return applyEraseUnnecessaryInputs(getOperation());
  if (decomposeConvolutions)
    return applyDecomposeConvolutions(getOperation());
  if (conv2DToImg2Col)
    return applyConv2DToImg2Col(getOperation());
}

namespace nelli {
void registerLinalgTransforms() { PassRegistration<LinalgTransforms>(); }
} // namespace nelli
