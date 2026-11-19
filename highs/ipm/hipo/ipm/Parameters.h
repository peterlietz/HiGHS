#ifndef HIPO_PARAMETERS_H
#define HIPO_PARAMETERS_H

#include "ipm/hipo/auxiliary/IntConfig.h"

namespace hipo {

// parameters for termination
const Int kMaxIterDefault = 200;
const double kIpmTolDefault = 1e-8;
const Int kMaxBadIter = 5;
// a step of at least kGapStallStep must reduce the gap of a feasible iterate
// by a factor kGapStallReduction, else it counts towards stagnation
const double kGapStallStep = 0.9;
const double kGapStallReduction = 0.9;
const double kDivergeTol = 1e3;
// Before stopping, HiGHS's KKT errors must be within tolerance, unless they
// stop decreasing: a check counts as progress if it reduces the smallest
// violation seen so far by this factor, and after this many checks without
// progress a violation within the factor 100 allowed by HiGHS is accepted.
const double kKktProgressFactor = 0.5;
const Int kKktChecksWithoutProgress = 2;

// parameters for correctors
const double kGammaCorrector = 0.1;
const double kSigmaAffine = 0.01;
const Int kMaxCorrectors = 10;
const double kMccIncreaseAlpha = 0.1;
const double kMccIncreaseMin = 0.1;
const double kSmallProduct = 1e-3;
const double kLargeProduct = 1e3;

// parameters for choice of AS or NE
const double kSystemSpopsWeight = 30.0;
const double kSystemRatioOpsThresh = 10.0;
const double kSystemRatioSnThresh = 1.5;
const double kSystemSymbNzMult = 5.0;

// parameters for choice of parallelism
const double kParallelLargeFlopsThresh = 1e7;
const double kParallelLargeSpeedupThresh = 1;
const double kParallelLargeSnThresh = 20.0;
const double kParallelSmallSnThresh = 5.0;
const Int kParallelMinNumberSn = 10;
const double kParallelLargeStorageGB = 20.0;
const double kParallelMaxTreeDepth = 1000;

// parameters for choice of ordering
const double kOrderingFlopsThresh = 1.2;

// parameters for choice of factorisation
const double kUplookFlopsThresh = 1e6;
const double kUplookNzPerColLower = 10;
const double kUplookNzPerColUpper = 25;
const double kUplookSpopsRatioLower = 20;
const double kUplookSpopsRatioUpper = 100;

// parameters for skipping AS or NE
const double kSkipSystemNzBoundsRatio = 50.0;

// parameters for leaving a network LP to IPX: the factorisation is predicted to
// need more than this many flops per nonzero of A, and more than this many
// flops in total
const double kIpxPreferredFlopsPerNz = 1e5;
const double kIpxPreferredFlopsMin = 1e9;

// parameters for iterative refinement
const Int kRefineMaxIter = 3;
const double kRefineTol = 1e-12;

// parameters for scaling
const double kScalingSmallCoeff = 1e-4;
const double kScalingLargeCoeff = 1e4;
const double kScalingSmallBoundDiff = 1e-3;

// parameters for free variables
const double kFreeVarsInitialBound = 1e4;
const double kFreeVarsCloseRatio = 0.5;
const double kFreeVarsMoveFactor = 10.0;

// parameters for parallel NE
const Int kParallelNEStructTasks = 50;   // 32 < . <= 64
const Int kParallelNEValuesTasks = 100;  // 64 < . <= 128
const Int kParallelNEnzPerColThresh = 10;
const Int kParallelNEnzPerRowThresh = 30;
const Int kParallelNEsizeThresh = 1e4;

// parameters for parallel solve
const double kParallelSolveMinSize = 1e4;
const double kParallelForwardMinSpeedup = 2;
const double kParallelBackwardMinSpeedup = 1.2;

// static regularisation
struct Regularisation {
  double primal = 1e-12;
  double dual = 1e-10;
};

}  // namespace hipo

#endif