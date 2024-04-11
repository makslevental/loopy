// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TRANSFORM_STRUCTUREDTRANSFORMOPSEXT_H
#define TRANSFORM_STRUCTUREDTRANSFORMOPSEXT_H

#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace linalg {
class LinalgOp;
} // namespace linalg
namespace pdl {
class FoOperationTyperOp;
} // namespace pdl
namespace scf {
class ForOp;
} // namespace scf
namespace transform {
class MatchCallbackOp;
} // namespace transform

/// Matches a C++ callback previously registered under `callbackName` and
/// taking arguments `args`.
/// Unpacks a number of handles `N` (asserts there are exactly `N` matched
/// ops but this could be relaxed if needed). Returns the tuple of handles.
template <int N, typename... MatchingArgs>
auto unpackRegisteredMatchCallback(ImplicitLocOpBuilder &b,
                                   StringRef callbackName,
                                   MatchingArgs... args) {
  SmallVector<Type> matchedTypes(N, pdl::OperationType::get(b.getContext()));
  auto matchOp = b.create<transform::MatchCallbackOp>(
      matchedTypes, callbackName, std::forward<decltype(args)>(args)...);
  assert(matchOp->getNumResults() == N && "Unexpected number of results");
  std::array<Value, N> a;
  for (int64_t i = 0; i < N; ++i)
    a[i] = matchOp->getResult(i);
  return std::tuple_cat(a);
}

} // namespace mlir

#define GET_OP_CLASSES
#include "StructuredTransformOpsExt.h.inc"

namespace mlir {

void registerTransformDialectStructuredExtension(DialectRegistry &registry);
void registerTransformDialectStructuredExtension(MLIRContext &context);

namespace transform {
class StructuredTransformOpsExtension
    : public mlir::transform::TransformDialectExtension<
          StructuredTransformOpsExtension> {
public:
  StructuredTransformOpsExtension();
};

} // namespace transform
} // namespace mlir

#endif // TRANSFORM_STRUCTUREDTRANSFORMOPSEXT_H
