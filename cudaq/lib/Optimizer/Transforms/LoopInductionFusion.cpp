/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "LoopAnalysis.h"
#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_LOOPINDUCTIONFUSION
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "cc-loop-induction-fusion"

using namespace mlir;

/// \file
/// For each normalized `cc.loop` that carries secondary induction variables
/// (extra loop args that step by a loop-invariant amount on every iteration),
/// replace uses of those args in the loop body with closed-form expressions in
/// terms of the loop control variable `i`:
///
///   j(i) = j_initial + i * step_j   (stepIsAdd)
///   j(i) = j_initial - i * step_j   (!stepIsAdd)
///
/// Simultaneously replace each secondary loop-result use in the parent scope
/// with the same closed-form evaluated at the loop control result, then
/// rebuild the cc.loop with the secondary args removed from its carried-value
/// set entirely.

namespace {

/// Replace secondary inductions with closed-form expressions of the loop
/// control variable, then rebuild the `cc.loop` without those secondary args in
/// its carried-value set.
struct FuseSecondaryInductions : public OpRewritePattern<cudaq::cc::LoopOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::cc::LoopOp loop,
                                PatternRewriter &rewriter) const override {
    // Fusion requires the loop control variable to be normalized so that j(i) =
    // j_initial ± i * step_j is a correct closed form.
    cudaq::opt::LoopComponents lcv;
    if (!cudaq::opt::isaInvariantLoop(loop, /*allowClosedInterval=*/false,
                                      /*allowEarlyExit=*/false, &lcv))
      return failure();
    if (!lcv.induction.has_value())
      return failure();

    auto secondaries = cudaq::opt::getSecondaryInductions(loop, lcv);
    if (secondaries.empty())
      return failure();

    auto loc = loop.getLoc();
    unsigned lcvIdx = *lcv.induction;

    // -----------------------------------------------------------------------
    // Phase 1 — body: replace each secondary's block arg with
    //   j(i) = j_initial ± body_lcv_arg * step_j
    // -----------------------------------------------------------------------
    Block &bodyEntry = loop.getBodyRegion().front();
    Value bodyLcv = bodyEntry.getArgument(lcvIdx);

    rewriter.setInsertionPointToStart(&bodyEntry);
    for (auto &si : secondaries) {
      Value initVal = si.initialValue;
      Value step = si.stepValue;
      Type ty = initVal.getType();
      if (step.getType() != ty)
        step = arith::ExtUIOp::create(rewriter, loc, ty, step);
      Value primCast = bodyLcv;
      if (primCast.getType() != ty)
        primCast = arith::ExtUIOp::create(rewriter, loc, ty, primCast);
      Value scaled = arith::MulIOp::create(rewriter, loc, primCast, step);
      Value jValue = si.stepIsAdd
                         ? arith::AddIOp::create(rewriter, loc, initVal, scaled)
                               .getResult()
                         : arith::SubIOp::create(rewriter, loc, initVal, scaled)
                               .getResult();
      bodyEntry.getArgument(si.argIndex).replaceAllUsesWith(jValue);
    }

    // -----------------------------------------------------------------------
    // Phase 2 — else region: same closed-form using the else block's lcv.
    // -----------------------------------------------------------------------
    if (loop.hasPythonElse()) {
      Block &elseEntry = loop.getElseRegion().front();
      Value elseLcv = elseEntry.getArgument(lcvIdx);
      rewriter.setInsertionPointToStart(&elseEntry);
      for (auto &si : secondaries) {
        Value initVal = si.initialValue;
        Value step = si.stepValue;
        Type ty = initVal.getType();
        if (step.getType() != ty)
          step = arith::ExtUIOp::create(rewriter, loc, ty, step);
        Value primCast = elseLcv;
        if (primCast.getType() != ty)
          primCast = arith::ExtUIOp::create(rewriter, loc, ty, primCast);
        Value scaled = arith::MulIOp::create(rewriter, loc, primCast, step);
        Value jElse =
            si.stepIsAdd ? arith::AddIOp::create(rewriter, loc, initVal, scaled)
                               .getResult()
                         : arith::SubIOp::create(rewriter, loc, initVal, scaled)
                               .getResult();
        elseEntry.getArgument(si.argIndex).replaceAllUsesWith(jElse);
      }
    }

    // -----------------------------------------------------------------------
    // Phase 3 — post-loop: replace each secondary result with
    //   j_final = j_initial ± lcv_result * step_j
    // -----------------------------------------------------------------------
    rewriter.setInsertionPointAfter(loop);
    Value lcvResult = loop.getResult(lcvIdx);
    for (auto &si : secondaries) {
      Value initVal = si.initialValue;
      Value step = si.stepValue;
      Type ty = initVal.getType();
      if (step.getType() != ty)
        step = arith::ExtUIOp::create(rewriter, loc, ty, step);
      Value primCast = lcvResult;
      if (primCast.getType() != ty)
        primCast = arith::ExtUIOp::create(rewriter, loc, ty, primCast);
      Value scaled = arith::MulIOp::create(rewriter, loc, primCast, step);
      Value jFinal = si.stepIsAdd
                         ? arith::AddIOp::create(rewriter, loc, initVal, scaled)
                               .getResult()
                         : arith::SubIOp::create(rewriter, loc, initVal, scaled)
                               .getResult();
      loop.getResult(si.argIndex).replaceAllUsesWith(jFinal);
    }

    // -----------------------------------------------------------------------
    // Phase 4 — build the new loop: remove the fused secondaries from the
    // loop's carried-values set so the loop is left in a clean form.
    //
    // Steps:
    //   a. Save the dead step computations before they lose their users.
    //   b. Strip fused positions from cc.condition and cc.continue terminators.
    //   c. Erase the dead step computations.
    //   d. Erase the fused block args from every region entry block.
    //   e. Create a new cc.loop with fewer initial args / result types.
    //   f. Steal the modified regions into the new loop.
    //   g. Redirect live results and erase the old loop.
    // -----------------------------------------------------------------------
    DenseSet<unsigned> fusedSet;
    for (auto &si : secondaries)
      fusedSet.insert(si.argIndex);

    // Returns a copy of `operands` with fused positions removed.
    auto keepLive = [&](OperandRange operands) -> SmallVector<Value> {
      SmallVector<Value> kept;
      for (unsigned i = 0, n = operands.size(); i < n; ++i)
        if (!fusedSet.count(i))
          kept.push_back(operands[i]);
      return kept;
    };

    // (a) Capture the step-computation values that will become dead once the
    //     step cc.continue drops them.
    SmallVector<Value> deadStepVals;
    if (loop.hasStep()) {
      auto stepCont =
          dyn_cast<cudaq::cc::ContinueOp>(loop.getStepRegion().front().back());
      if (stepCont)
        for (auto &si : secondaries)
          deadStepVals.push_back(stepCont.getOperands()[si.argIndex]);
    }

    // (b) Fix terminators.
    // IMPORTANT: Phase 3 left the insertion point after the loop in the parent
    // block.  replaceOpWithNewOp uses the current insertion point for the
    // newly created op, so we must explicitly reset it to just before the op
    // being replaced each time, then restore it afterwards.
    //
    //   While region: cc.condition($cond, $results...) — drop fused $results.
    for (auto &block : loop.getWhileRegion())
      if (block.hasNoSuccessors())
        if (auto cond = dyn_cast<cudaq::cc::ConditionOp>(block.back())) {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(cond);
          rewriter.replaceOpWithNewOp<cudaq::cc::ConditionOp>(
              cond, cond.getCondition(), keepLive(cond.getResults()));
        }

    //   Body, step, else: cc.continue($operands...) — drop fused positions.
    auto fixContinues = [&](Region &reg) {
      for (auto &block : reg)
        if (block.hasNoSuccessors())
          if (auto cont = dyn_cast<cudaq::cc::ContinueOp>(block.back())) {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPoint(cont);
            rewriter.replaceOpWithNewOp<cudaq::cc::ContinueOp>(
                cont, keepLive(cont.getOperands()));
          }
    };
    fixContinues(loop.getBodyRegion());
    if (loop.hasStep())
      fixContinues(loop.getStepRegion());
    if (loop.hasPythonElse())
      fixContinues(loop.getElseRegion());

    // (c) Erase the now-dead step computations (the addi/subi ops that
    //     stepped the secondary; they have no users after (b)).
    for (Value v : deadStepVals)
      if (v && v.use_empty())
        if (auto *defOp = v.getDefiningOp())
          rewriter.eraseOp(defOp);

    // (d) Erase the fused block args from every region's entry block.
    //     Process in descending index order so earlier indices stay valid.
    SmallVector<unsigned> deadDesc(fusedSet.begin(), fusedSet.end());
    llvm::sort(deadDesc, std::greater<unsigned>());
    for (auto *reg : loop.getRegions()) {
      if (reg->empty())
        continue;
      for (unsigned idx : deadDesc)
        reg->front().eraseArgument(idx);
    }

    // (e) Assemble new initial args and result types.
    SmallVector<Value> newInitArgs;
    SmallVector<Type> newResultTypes;
    for (unsigned i = 0, n = loop.getNumResults(); i < n; ++i)
      if (!fusedSet.count(i)) {
        newInitArgs.push_back(loop.getInitialArgs()[i]);
        newResultTypes.push_back(loop.getResultTypes()[i]);
      }

    // Create the new loop with empty region builders; we'll move the
    // already-modified regions in below.
    rewriter.setInsertionPoint(loop);
    auto newLoop = cudaq::cc::LoopOp::create(
        rewriter, loop.getLoc(), newResultTypes, newInitArgs,
        loop.isPostConditional(), [](OpBuilder &, Location, Region &) {},
        [](OpBuilder &, Location, Region &) {},
        /*stepBuilder=*/nullptr);

    // (f) Propagate attributes and set normalized on the new loop.
    newLoop->setDiscardableAttrs(loop->getDiscardableAttrDictionary());
    newLoop->setAttr(cudaq::opt::NormalizedLoopAttr,
                     UnitAttr::get(rewriter.getContext()));

    // Move the modified regions into the new loop.
    newLoop.getWhileRegion().takeBody(loop.getWhileRegion());
    newLoop.getBodyRegion().takeBody(loop.getBodyRegion());
    newLoop.getStepRegion().takeBody(loop.getStepRegion());
    newLoop.getElseRegion().takeBody(loop.getElseRegion());

    // (g) Redirect live results from old loop to new loop, then erase old.
    unsigned newIdx = 0;
    for (unsigned i = 0, n = loop.getNumResults(); i < n; ++i)
      if (!fusedSet.count(i))
        loop.getResult(i).replaceAllUsesWith(newLoop.getResult(newIdx++));

    rewriter.eraseOp(loop);
    return success();
  }
};

struct LoopInductionFusionPass
    : public cudaq::opt::impl::LoopInductionFusionBase<
          LoopInductionFusionPass> {
  using LoopInductionFusionBase::LoopInductionFusionBase;

  void runOnOperation() override {
    auto *op = getOperation();
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<FuseSecondaryInductions>(ctx);
    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      op->emitOpError("loop induction fusion failed");
      signalPassFailure();
    }
  }
};

} // namespace
