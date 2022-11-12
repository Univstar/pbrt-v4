// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0
//
// This file is maintained by PKU-VCL Lab.

#include <pbrt/shapes.h>

#include <pbrt/textures.h>
#ifdef PBRT_BUILD_GPU_RENDERER
#include <pbrt/gpu/util.h>
#endif  // PBRT_BUILD_GPU_RENDERER
#include <pbrt/interaction.h>
#include <pbrt/options.h>
#include <pbrt/paramdict.h>
#include <pbrt/util/check.h>
#include <pbrt/util/error.h>
#include <pbrt/util/file.h>
#include <pbrt/util/float.h>
#include <pbrt/util/image.h>
#include <pbrt/util/loopsubdiv.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/math.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/print.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/splines.h>
#include <pbrt/util/stats.h>

#if defined(PBRT_BUILD_GPU_RENDERER)
#include <cuda.h>
#endif

#include <algorithm>

namespace pbrt {

BicubicPatch *BicubicPatch::Create(const Transform *renderFromObject,
                                         bool reverseOrientation,
                                         const ParameterDictionary &parameters,
                                         const FileLoc *loc, Allocator alloc) {
	std::vector<Point3f> P = parameters.GetPoint3fArray("P");
	if (P.size() != 16) {
		Error(loc, "Number of vertex positions \"P\" for bicubic patch must be 16.");
		return {};
	}
	std::array<Point3f, 16> points;
	for (size_t i = 0; i < 16; ++i) points[i] = P[i];

	return alloc.new_object<BicubicPatch>(renderFromObject, reverseOrientation, points);
}

std::string BicubicPatch::ToString() const {
	return StringPrintf("[ BicubicPatch reverseOrientation: %s transformSwapsHandedness: %s "
						"points[0]: %s points[4]: %s "
						"points[8]: %s points[12]: %s",
						reverseOrientation, transformSwapsHandedness,
						points[0], points[4], points[8], points[12]);
}

BicubicPatch::BicubicPatch(const Transform *renderFromObject, bool reverseOrientation,
                           const pstd::array<Point3f, 16> &points)
	: reverseOrientation(reverseOrientation),
	  transformSwapsHandedness(renderFromObject->SwapsHandedness()),
	  points(points) {
	for (Point3f &pt : this->points)
		pt = (*renderFromObject)(pt);
}

Bounds3f BicubicPatch::Bounds() const {
	Bounds3f bounds = Bounds3f(points[0], points[1]);
	for (size_t i = 2; i < 16; i++) bounds = Union(bounds, points[i]);
	return bounds;
}

DirectionCone BicubicPatch::NormalBounds() const {
}

}
