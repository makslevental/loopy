// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "ListenerCSE.h"

#include <deque>

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/IR/OpDefinition.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/RecyclingAllocator.h"

using llvm::dbgs;
#define DEBUG_TYPE "listener-cse"
#define DBGS() (dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(dbgs() << '[' << DEBUG_TYPE << "] " << X)

using namespace mlir;

//===----------------------------------------------------------------------===//
// BEGIN copied from mlir/lib/Transforms/CSE.cpp
//===----------------------------------------------------------------------===//
namespace {
struct SimpleOperationInfo : public llvm::DenseMapInfo<Operation *> {
  static unsigned getHashValue(const Operation *opC) {
    return OperationEquivalence::computeHash(
        const_cast<Operation *>(opC),
        /*hashOperands=*/OperationEquivalence::directHashValue,
        /*hashResults=*/OperationEquivalence::ignoreHashValue,
        OperationEquivalence::IgnoreLocations);
  }
  static bool isEqual(const Operation *lhsC, const Operation *rhsC) {
    auto *lhs = const_cast<Operation *>(lhsC);
    auto *rhs = const_cast<Operation *>(rhsC);
    if (lhs == rhs)
      return true;
    if (lhs == getTombstoneKey() || lhs == getEmptyKey() ||
        rhs == getTombstoneKey() || rhs == getEmptyKey())
      return false;

    // If op has no regions, operation equivalence w.r.t operands alone is
    // enough.
    if (lhs->getNumRegions() == 0 && rhs->getNumRegions() == 0) {
      return OperationEquivalence::isEquivalentTo(
          const_cast<Operation *>(lhsC), const_cast<Operation *>(rhsC),
          OperationEquivalence::exactValueMatch,
          /*markEquivalent=*/nullptr, OperationEquivalence::IgnoreLocations);
    }

    // If lhs or rhs does not have a single region with a single block, they
    // aren't CSEed for now.
    if (lhs->getNumRegions() != 1 || rhs->getNumRegions() != 1 ||
        !llvm::hasSingleElement(lhs->getRegion(0)) ||
        !llvm::hasSingleElement(rhs->getRegion(0)))
      return false;

    // Compare the two blocks.
    Block &lhsBlock = lhs->getRegion(0).front();
    Block &rhsBlock = rhs->getRegion(0).front();

    // Don't CSE if number of arguments differ.
    if (lhsBlock.getNumArguments() != rhsBlock.getNumArguments())
      return false;

    // Map to store `Value`s from `lhsBlock` that are equivalent to `Value`s
    // in `rhsBlock`. `Value`s from `lhsBlock` are the key.
    DenseMap<Value, Value> areEquivalentValues;
    for (auto bbArgs : llvm::zip(lhs->getRegion(0).getArguments(),
                                 rhs->getRegion(0).getArguments())) {
      areEquivalentValues[std::get<0>(bbArgs)] = std::get<1>(bbArgs);
    }

    // Helper function to get the parent operation.
    auto getParent = [](Value v) -> Operation * {
      if (auto blockArg = v.dyn_cast<BlockArgument>())
        return blockArg.getParentBlock()->getParentOp();
      return v.getDefiningOp()->getParentOp();
    };

    // Callback to compare if operands of ops in the region of `lhs` and `rhs`
    // are equivalent.
    auto checkEquivalent = [&](Value lhsValue,
                               Value rhsValue) -> LogicalResult {
      if (lhsValue == rhsValue)
        return success();
      if (areEquivalentValues.lookup(lhsValue) == rhsValue)
        return success();
      return failure();
    };

    // Callback to compare if results of ops in the region of `lhs` and `rhs`
    // are equivalent.
    auto markEquivalent = [&](Value lhsResult, Value rhsResult) {
      if (getParent(lhsResult) == lhs && getParent(rhsResult) == rhs) {
        areEquivalentValues.insert({lhsResult, rhsResult});
      }
    };

    return OperationEquivalence::isEquivalentTo(
        const_cast<Operation *>(lhsC), const_cast<Operation *>(rhsC),
        checkEquivalent, markEquivalent, OperationEquivalence::IgnoreLocations);
  }
};
} // namespace

namespace {
/// Simple common sub-expression elimination.
//===----------------------------------------------------------------------===//
// END copied from mlir/lib/Transforms/CSE.cpp
//===----------------------------------------------------------------------===//
/// Copy of CSE::runOnOperation, without the pass baggage.
// struct CSE : public impl::CSEBase<CSE> {
struct CSE {
  //===----------------------------------------------------------------------===//
  // BEGIN copied from mlir/lib/Transforms/CSE.cpp
  //===----------------------------------------------------------------------===//
  /// Shared implementation of operation elimination and scoped map
  /// definitions.
  using AllocatorTy = llvm::RecyclingAllocator<
      llvm::BumpPtrAllocator,
      llvm::ScopedHashTableVal<Operation *, Operation *>>;
  using ScopedMapTy = llvm::ScopedHashTable<Operation *, Operation *,
                                            SimpleOperationInfo, AllocatorTy>;

  /// Cache holding MemoryEffects information between two operations. The
  /// first operation is stored has the key. The second operation is stored
  /// inside a pair in the value. The pair also hold the MemoryEffects between
  /// those two operations. If the MemoryEffects is nullptr then we assume
  /// there is no operation with MemoryEffects::Write between the two
  /// operations.
  using MemEffectsCache =
      DenseMap<Operation *, std::pair<Operation *, MemoryEffects::Effect *>>;

  /// Represents a single entry in the depth first traversal of a CFG.
  struct CFGStackNode {
    CFGStackNode(ScopedMapTy &knownValues, DominanceInfoNode *node)
        : scope(knownValues), node(node), childIterator(node->begin()) {}

    /// Scope for the known values.
    ScopedMapTy::ScopeTy scope;

    DominanceInfoNode *node;
    DominanceInfoNode::const_iterator childIterator;

    /// If this node has been fully processed yet or not.
    bool processed = false;
  };

  /// Attempt to eliminate a redundant operation. Returns success if the
  /// operation was marked for removal, failure otherwise.
  LogicalResult simplifyOperation(ScopedMapTy &knownValues, Operation *op,
                                  bool hasSSADominance);
  void simplifyBlock(ScopedMapTy &knownValues, Block *bb, bool hasSSADominance);
  void simplifyRegion(ScopedMapTy &knownValues, Region &region);

  // void runOnOperation() override;
  void doItOnOperation(Operation *rootOp, DominanceInfo *domInfo,
                       RewriterBase::Listener *listener);

private:
  void replaceUsesAndDelete(ScopedMapTy &knownValues, Operation *op,
                            Operation *existing, bool hasSSADominance);

  /// Check if there is side-effecting operations other than the given effect
  /// between the two operations.
  bool hasOtherSideEffectingOpInBetween(Operation *fromOp, Operation *toOp);

  /// Operations marked as dead and to be erased.
  std::vector<Operation *> opsToErase;
  DominanceInfo *domInfo = nullptr;
  MemEffectsCache memEffectsCache;
  //===----------------------------------------------------------------------===//
  // END copied from mlir/lib/Transforms/CSE.cpp
  //===----------------------------------------------------------------------===//
  /// An optional listener to notify of replaced or erased operations.
  RewriterBase::Listener *listener;
  int64_t numDCE = 0, numCSE = 0;

  //===----------------------------------------------------------------------===//
  // BEGIN copied from mlir/lib/Transforms/CSE.cpp
  //===----------------------------------------------------------------------===//
};
} // namespace

void CSE::replaceUsesAndDelete(ScopedMapTy &knownValues, Operation *op,
                               Operation *existing, bool hasSSADominance) {
  // If we find one then replace all uses of the current operation with the
  // existing one and mark it for deletion. We can only replace an operand in
  // an operation if it has not been visited yet.
  if (hasSSADominance) {
    // If the region has SSA dominance, then we are guaranteed to have not
    // visited any use of the current operation.
    //===----------------------------------------------------------------------===//
    // END copied from mlir/lib/Transforms/CSE.cpp
    //===----------------------------------------------------------------------===//
    if (listener)
      listener->notifyOperationReplaced(op, existing->getResults());
    //===----------------------------------------------------------------------===//
    // BEGIN copied from mlir/lib/Transforms/CSE.cpp
    //===----------------------------------------------------------------------===//
    op->replaceAllUsesWith(existing);
    opsToErase.push_back(op);
  } else {
    // When the region does not have SSA dominance, we need to check if we
    // have visited a use before replacing any use.
    for (auto it : llvm::zip(op->getResults(), existing->getResults())) {
      std::get<0>(it).replaceUsesWithIf(
          std::get<1>(it), [&](OpOperand &operand) {
            return !knownValues.count(operand.getOwner());
          });
    }

    // There may be some remaining uses of the operation.
    if (op->use_empty())
      opsToErase.push_back(op);
  }

  // If the existing operation has an unknown location and the current
  // operation doesn't, then set the existing op's location to that of the
  // current op.
  if (existing->getLoc().isa<UnknownLoc>() && !op->getLoc().isa<UnknownLoc>())
    existing->setLoc(op->getLoc());

  ++numCSE;
}

bool CSE::hasOtherSideEffectingOpInBetween(Operation *fromOp, Operation *toOp) {
  assert(fromOp->getBlock() == toOp->getBlock());
  assert(
      isa<MemoryEffectOpInterface>(fromOp) &&
      cast<MemoryEffectOpInterface>(fromOp).hasEffect<MemoryEffects::Read>() &&
      isa<MemoryEffectOpInterface>(toOp) &&
      cast<MemoryEffectOpInterface>(toOp).hasEffect<MemoryEffects::Read>());
  Operation *nextOp = fromOp->getNextNode();
  auto result =
      memEffectsCache.try_emplace(fromOp, std::make_pair(fromOp, nullptr));
  if (result.second) {
    auto memEffectsCachePair = result.first->second;
    if (memEffectsCachePair.second == nullptr) {
      // No MemoryEffects::Write has been detected until the cached operation.
      // Continue looking from the cached operation to toOp.
      nextOp = memEffectsCachePair.first;
    } else {
      // MemoryEffects::Write has been detected before so there is no need to
      // check further.
      return true;
    }
  }
  while (nextOp && nextOp != toOp) {
    auto nextOpMemEffects = dyn_cast<MemoryEffectOpInterface>(nextOp);
    // TODO: Do we need to handle other effects generically?
    // If the operation does not implement the MemoryEffectOpInterface we
    // conservatively assumes it writes.
    if ((nextOpMemEffects &&
         nextOpMemEffects.hasEffect<MemoryEffects::Write>()) ||
        !nextOpMemEffects) {
      result.first->second =
          std::make_pair(nextOp, MemoryEffects::Write::get());
      return true;
    }
    nextOp = nextOp->getNextNode();
  }
  result.first->second = std::make_pair(toOp, nullptr);
  return false;
}

/// Attempt to eliminate a redundant operation.
LogicalResult CSE::simplifyOperation(ScopedMapTy &knownValues, Operation *op,
                                     bool hasSSADominance) {
  // Don't simplify terminator operations.
  if (op->hasTrait<OpTrait::IsTerminator>())
    return failure();

  // If the operation is already trivially dead just add it to the erase list.
  if (isOpTriviallyDead(op)) {
    opsToErase.push_back(op);
    ++numDCE;
    return success();
  }

  // Don't simplify operations with nested blocks. We don't currently model
  // equality comparisons correctly among other things. It is also unclear
  // whether we would want to CSE such operations.
  if (!(op->getNumRegions() == 0 ||
        (op->getNumRegions() == 1 && llvm::hasSingleElement(op->getRegion(0)))))
    return failure();

  // Some simple use case of operation with memory side-effect are dealt with
  // here. Operations with no side-effect are done after.
  if (!isMemoryEffectFree(op)) {
    auto memEffects = dyn_cast<MemoryEffectOpInterface>(op);
    // TODO: Only basic use case for operations with MemoryEffects::Read can
    // be eleminated now. More work needs to be done for more complicated
    // patterns and other side-effects.
    if (!memEffects || !memEffects.onlyHasEffect<MemoryEffects::Read>())
      return failure();

    // Look for an existing definition for the operation.
    if (auto *existing = knownValues.lookup(op)) {
      if (existing->getBlock() == op->getBlock() &&
          !hasOtherSideEffectingOpInBetween(existing, op)) {
        // The operation that can be deleted has been reach with no
        // side-effecting operations in between the existing operation and
        // this one so we can remove the duplicate.
        replaceUsesAndDelete(knownValues, op, existing, hasSSADominance);
        return success();
      }
    }
    knownValues.insert(op, op);
    return failure();
  }

  // Look for an existing definition for the operation.
  if (auto *existing = knownValues.lookup(op)) {
    replaceUsesAndDelete(knownValues, op, existing, hasSSADominance);
    ++numCSE;
    return success();
  }

  // Otherwise, we add this operation to the known values map.
  knownValues.insert(op, op);
  return failure();
}

void CSE::simplifyBlock(ScopedMapTy &knownValues, Block *bb,
                        bool hasSSADominance) {
  for (auto &op : *bb) {
    // If the operation is simplified, we don't process any held regions.
    if (succeeded(simplifyOperation(knownValues, &op, hasSSADominance)))
      continue;

    // Most operations don't have regions, so fast path that case.
    if (op.getNumRegions() == 0)
      continue;

    // If this operation is isolated above, we can't process nested regions
    // with the given 'knownValues' map. This would cause the insertion of
    // implicit captures in explicit capture only regions.
    if (op.mightHaveTrait<OpTrait::IsIsolatedFromAbove>()) {
      ScopedMapTy nestedKnownValues;
      for (auto &region : op.getRegions())
        simplifyRegion(nestedKnownValues, region);
      continue;
    }

    // Otherwise, process nested regions normally.
    for (auto &region : op.getRegions())
      simplifyRegion(knownValues, region);
  }
  // Clear the MemoryEffects cache since its usage is by block only.
  memEffectsCache.clear();
}

void CSE::simplifyRegion(ScopedMapTy &knownValues, Region &region) {
  // If the region is empty there is nothing to do.
  if (region.empty())
    return;

  bool hasSSADominance = domInfo->hasSSADominance(&region);

  // If the region only contains one block, then simplify it directly.
  if (region.hasOneBlock()) {
    ScopedMapTy::ScopeTy scope(knownValues);
    simplifyBlock(knownValues, &region.front(), hasSSADominance);
    return;
  }

  // If the region does not have dominanceInfo, then skip it.
  // TODO: Regions without SSA dominance should define a different
  // traversal order which is appropriate and can be used here.
  if (!hasSSADominance)
    return;

  // Note, deque is being used here because there was significant performance
  // gains over vector when the container becomes very large due to the
  // specific access patterns. If/when these performance issues are no
  // longer a problem we can change this to vector. For more information see
  // the llvm mailing list discussion on this:
  // http://lists.llvm.org/pipermail/llvm-commits/Week-of-Mon-20120116/135228.html
  std::deque<std::unique_ptr<CFGStackNode>> stack;

  // Process the nodes of the dom tree for this region.
  stack.emplace_back(std::make_unique<CFGStackNode>(
      knownValues, domInfo->getRootNode(&region)));

  while (!stack.empty()) {
    auto &currentNode = stack.back();

    // Check to see if we need to process this node.
    if (!currentNode->processed) {
      currentNode->processed = true;
      simplifyBlock(knownValues, currentNode->node->getBlock(),
                    hasSSADominance);
    }

    // Otherwise, check to see if we need to process a child node.
    if (currentNode->childIterator != currentNode->node->end()) {
      auto *childNode = *(currentNode->childIterator++);
      stack.emplace_back(
          std::make_unique<CFGStackNode>(knownValues, childNode));
    } else {
      // Finally, if the node and all of its children have been processed
      // then we delete the node.
      stack.pop_back();
    }
  }
}

//===----------------------------------------------------------------------===//
// END copied from mlir/lib/Transforms/CSE.cpp
//===----------------------------------------------------------------------===//

/// Copy of CSE::runOnOperation, without the pass baggage.
void CSE::doItOnOperation(Operation *rootOp, DominanceInfo *domInfo,
                          RewriterBase::Listener *listener) {
  /// A scoped hash table of defining operations within a region.
  ScopedMapTy knownValues;
  this->domInfo = domInfo;
  this->listener = listener;

  for (auto &region : rootOp->getRegions())
    simplifyRegion(knownValues, region);

  /// Erase any operations that were marked as dead during simplification.
  for (auto *op : opsToErase) {
    if (listener)
      listener->notifyOperationRemoved(op);
    op->erase();
  }
  opsToErase.clear();
}

/// Run CSE on the provided operation
LogicalResult
mlir::eliminateCommonSubexpressions(Operation *op, DominanceInfo *domInfo,
                                    RewriterBase::Listener *listener) {
  //  assert(op->hasTrait<OpTrait::IsIsolatedFromAbove>() &&
  //         "can only do CSE on isolated-from-above ops");
  std::optional<DominanceInfo> defaultDomInfo;
  if (domInfo == nullptr) {
    defaultDomInfo.emplace(op);
    domInfo = &*defaultDomInfo;
  }
  CSE().doItOnOperation(op, domInfo, listener);
  return success();
}

//===----------------------------------------------------------------------===//
// TrackingListener
//===----------------------------------------------------------------------===//

/// Find the linalg op that defines all values in range, potentially
/// transitively through tensor casts.
static FailureOr<linalg::LinalgOp>
findSingleLinalgOpDefiningAll(ValueRange range) {
  LLVM_DEBUG(DBGS() << "Start findSingleLinalgOpDefiningAll\n");
  linalg::LinalgOp sourceOp = nullptr;
  for (Value value : range) {
    // See through tensor casts and reshape ops.
    //
    // TODO: we may need some generalization (interfaces?) of this for other
    // operations, especially multi-operand ones to understand which of their
    // operands may be coming from a Linalg op. Or a completely different
    // mechanism of tracking op replacement at creation, or even different
    // patterns that identify the "main" result of a transformation.
    while (isa<tensor::CastOp, tensor::CollapseShapeOp, tensor::ExpandShapeOp,
               tensor::InsertSliceOp>(value.getDefiningOp())) {
      value = llvm::TypeSwitch<Operation *, Value>(value.getDefiningOp())
                  .Case([](tensor::CastOp op) { return op.getSource(); })
                  .Case([](tensor::CollapseShapeOp op) { return op.getSrc(); })
                  .Case([](tensor::ExpandShapeOp op) { return op.getSrc(); })
                  .Case([](tensor::InsertSliceOp op) { return op.getSource(); })
                  .Default([](Operation *) {
                    llvm_unreachable("Wrong op type");
                    return Value();
                  });
    }

    if (auto currentSourceOp = value.getDefiningOp<linalg::LinalgOp>()) {
      if (!sourceOp || sourceOp == currentSourceOp) {
        sourceOp = currentSourceOp;
        continue;
      }

      LLVM_DEBUG(
          DBGS() << "--different source linalg ops for replacing one op: \n"
                 << sourceOp << "\n"
                 << currentSourceOp << "\n");
      return failure();
    }
    LLVM_DEBUG(DBGS() << "--replacing linalg op with unknown non-linalg op:\n"
                      << *value.getDefiningOp() << "\n");
    return failure();
  }
  return sourceOp;
}

/// Find the scf "for" op that defines all values in the range.
/// Take into account the the op may just disappear when it is replaced by its
/// body, in the case od a single iteration loop.
// It is unclear atm how to account for this properly.
static FailureOr<Operation *> findSingleForOpDefiningAll(ValueRange range) {
  LLVM_DEBUG(DBGS() << "Start findSingleForOpDefiningAll\n");
  Operation *forOp = nullptr;
  for (Value value : range) {
    LLVM_DEBUG(DBGS() << "--find for: " << value << "\n");
    // Block arguments are just dropped.
    auto currentSourceOp = value.getDefiningOp();
    if (!currentSourceOp) {
      LLVM_DEBUG(DBGS() << "--replacing tracked op with bbarg -> SKIP\n");
      continue;
    }
    auto currentForOp = dyn_cast<scf::ForOp>(currentSourceOp);
    if (!forOp || (currentForOp && forOp == currentForOp)) {
      forOp = currentSourceOp;
      continue;
    }
    LLVM_DEBUG(DBGS() << "---no single scf.for replacement found -> SKIP\n");
    LLVM_DEBUG(
        DBGS() << "---WARNING: this will drop tracking of the scf.for\n");
    return nullptr;
  }
  return forOp;
}

/// Find the op that defines all values in the range.
static FailureOr<Operation *> findSingleOpDefiningAll(ValueRange range) {
  Operation *op = nullptr;
  for (Value value : range) {
    // Block arguments are just dropped.
    auto currentSourceOp = value.getDefiningOp();
    if (!currentSourceOp) {
      LLVM_DEBUG(DBGS() << "replacing tracked op with bbarg\n");
      continue;
    }

    if (!op || op == currentSourceOp) {
      op = currentSourceOp;
      continue;
    }

    LLVM_DEBUG(DBGS() << "different source op when replacing one op\n");
    return failure();
  }
  return op;
}

// Find a single op that defines all values in the range, optionally
// transitively through other operations in an op-specific way.
static FailureOr<Operation *> findSingleDefiningOp(Operation *replacedOp,
                                                   ValueRange range) {
  return llvm::TypeSwitch<Operation *, FailureOr<Operation *>>(replacedOp)
      .Case<linalg::LinalgOp>([&](linalg::LinalgOp) -> FailureOr<Operation *> {
        auto op = findSingleLinalgOpDefiningAll(range);
        if (failed(op))
          return failure();
        return op->getOperation();
      })
      .Case<scf::ForOp>([&](scf::ForOp) -> FailureOr<Operation *> {
        return findSingleForOpDefiningAll(range);
      })
      .Default([&](Operation *) -> FailureOr<Operation *> {
        return findSingleOpDefiningAll(range);
      });
}

void mlir::TrackingListener::notifyOperationReplaced(Operation *op,
                                                     ValueRange newValues) {
  // Bail out if in error state.
  if (hadErrors)
    return;

  // Exit early if the op is not tracked.
  SmallVector<Value> handles;
  if (failed(getTransformState().getHandlesForPayloadOp(op, handles))) {
    LLVM_DEBUG(DBGS() << "no tracking handle to remove\n");
    return;
  }

  FailureOr<Operation *> replacement = findSingleDefiningOp(op, newValues);
  if (failed(replacement)) {
    LLVM_DEBUG(DBGS() << "could not find replacement for tracked op\n");
    emitError(op) << "could not find replacement for tracked op";
    return;
  }

  // If this would cause an error with replacement, drop instead.
  if (*replacement && (*replacement)->getNumResults() != op->getNumResults()) {
    LLVM_DEBUG(DBGS() << "failsafe error tracking activated due to mismatched "
                         "number of results for op: "
                      << op << " and replacement " << *replacement << "\n");
    replacement = nullptr;
  }

  if (*replacement == nullptr) {
    // TODO: Check if the handle is dead. Otherwise, the op should not be
    // dropped. This needs a change in the transform dialect interpreter.
    LLVM_DEBUG(DBGS() << "removing tracked @" << op << " : " << *op << "\n");
  } else {
    LLVM_DEBUG(DBGS() << "replacing tracked @" << op << " : " << *op << " with "
                      << **replacement << "\n");
  }
  mayFail(replacePayloadOp(op, *replacement));
}

void mlir::TrackingListener::notifyOperationRemoved(Operation *op) {
  // Bail out if in error state.
  if (hadErrors)
    return;

  // TODO: Walk can be removed when D144193 has landed.
  op->walk([&](Operation *op) {
    // Exit early if the op is not tracked.
    SmallVector<Value> handles;
    if (failed(getTransformState().getHandlesForPayloadOp(op, handles))) {
      LLVM_DEBUG(DBGS() << "no tracking handle to remove\n");
      return;
    }
    LLVM_DEBUG(DBGS() << "removing tracked @" << op << " : " << *op << "\n");
    mayFail(replacePayloadOp(op, nullptr));
  });
}

void mlir::TrackingListener::removeMappings(Operation *op) {
  // Bail out if in error state.
  if (hadErrors)
    return;

  // Replacing the tracked op with null will stop the tracking.
  LLVM_DEBUG(DBGS() << "removing mappings @" << op << " : " << *op << "\n");
  mayFail(replacePayloadOp(op, nullptr));
}

namespace mlir {
namespace transform {
#ifndef NDEBUG
namespace detail {
void checkImplementsTransformOpInterface(StringRef name, MLIRContext *context) {
  // Since the operation is being inserted into the Transform dialect and the
  // dialect does not implement the interface fallback, only check for the op
  // itself having the interface implementation.
  RegisteredOperationName opName =
      *RegisteredOperationName::lookup(name, context);
  assert((opName.hasInterface<TransformOpInterface>() ||
          opName.hasTrait<OpTrait::IsTerminator>()) &&
         "non-terminator ops injected into the transform dialect must "
         "implement TransformOpInterface");
  assert(opName.hasInterface<MemoryEffectOpInterface>() &&
         "ops injected into the transform dialect must implement "
         "MemoryEffectsOpInterface");
}

void checkImplementsTransformHandleTypeInterface(TypeID typeID,
                                                 MLIRContext *context) {
  const auto &abstractType = AbstractType::lookup(typeID, context);
  assert((abstractType.hasInterface(
              TransformHandleTypeInterface::getInterfaceID()) ||
          abstractType.hasInterface(
              TransformParamTypeInterface::getInterfaceID()) ||
          abstractType.hasInterface(
              TransformValueHandleTypeInterface::getInterfaceID())) &&
         "expected Transform dialect type to implement one of the three "
         "interfaces");
}
} // namespace detail
#endif // NDEBUG
} // namespace transform
} // namespace mlir
