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
#include <queue>

namespace pbrt {

STAT_PERCENT("Intersections/Ray-bicubic patch intersection tests", nBcpHits, nBcpTests);
STAT_COUNTER("Geometry/BicubicPatches", nBcps);
STAT_COUNTER("Geometry/Split bicubic patches", nSplitBcps);

BicubicPatch *BicubicPatch::Create(const Transform *renderFromObject,
                                         bool reverseOrientation,
                                         const ParameterDictionary &parameters,
                                         const FileLoc *loc, Allocator alloc) {
	std::vector<Point3f> P = parameters.GetPoint3fArray("P");
	std::vector<Point2f> uvs = parameters.GetPoint2fArray("uv");

	if (P.size() != 16) {
		Error(loc, "Number of vertex positions \"P\" for bicubic patch must be 16.");
		return {};
	}
	if (uvs.empty()) {
		uvs = {Point2f(0, 0), Point2f(1, 1)};
	} else if (uvs.size() != 2) {
        Error(loc, "Number of \"uv\"s for bicubic patch must be 2. "
                   "Discarding uvs.");
		uvs = {Point2f(0, 0), Point2f(1, 1)};
	}

	pstd::array<Point3f, 16> points;
	for (size_t i = 0; i < 16; ++i) points[i] = P[i];

	return alloc.new_object<BicubicPatch>(renderFromObject, reverseOrientation, points, uvs[0], uvs[1]);
}

std::string BicubicPatch::ToString() const {
	return StringPrintf("[ BicubicPatch reverseOrientation: %s transformSwapsHandedness: %s "
						"points[0]: %s points[4]: %s "
						"points[8]: %s points[12]: %s",
						reverseOrientation, transformSwapsHandedness,
						points[0], points[4], points[8], points[12]);
}

BicubicPatch::BicubicPatch(const Transform *renderFromObject, bool reverseOrientation,
                           const pstd::array<Point3f, 16> &points,
  			               const Point2f &uvMin, const Point2f &uvMax)
	: reverseOrientation(reverseOrientation),
	  transformSwapsHandedness(renderFromObject->SwapsHandedness()),
	  points(points),
	  uvMin(uvMin),
	  uvMax(uvMax) {
	for (Point3f &pt : this->points)
		pt = (*renderFromObject)(pt);
	++nBcps;
}

Bounds3f BicubicPatch::Bounds() const {
	Bounds3f bounds = Bounds3f(points[0], points[1]);
	for (size_t i = 2; i < 16; i++) bounds = Union(bounds, points[i]);
	return bounds;
}

DirectionCone BicubicPatch::NormalBounds() const {
	return DirectionCone::EntireSphere();
}

pstd::optional<ShapeIntersection> BicubicPatch::Intersect(const Ray &ray, Float tMax) const {
	pstd::optional<ShapeIntersection> si;
	IntersectRay(ray, tMax, &si);
	return si;
}

bool BicubicPatch::IntersectP(const Ray &ray, Float tMax) const {
	return IntersectRay(ray, tMax, nullptr);
}

Float BicubicPatch::Area() const {
	// TODO
    return 0;
}

pstd::optional<ShapeSample> BicubicPatch::Sample(Point2f u) const {
    LOG_FATAL("BicubicPatch::Sample not implemented.");
    return {};
}

Float BicubicPatch::PDF(const Interaction &) const {
    LOG_FATAL("BicubicPatch::PDF not implemented.");
    return {};
}

pstd::optional<ShapeSample> BicubicPatch::Sample(const ShapeSampleContext &ctx,
                                          Point2f u) const {
    LOG_FATAL("BicubicPatch::Sample not implemented.");
    return {};
}

Float BicubicPatch::PDF(const ShapeSampleContext &ctx, Vector3f wi) const {
    LOG_FATAL("BicubicPatch::PDF not implemented.");
    return {};
}

bool BicubicPatch::IntersectRay(const Ray &ray, Float tMax,
								pstd::optional<ShapeIntersection> *si) const {
#ifndef PBRT_IS_GPU_CODE
	++nBcpTests;
#endif
	// Project bicubic patch control points to plane perpendicular to ray
	Vector3f dx = Cross(ray.d, points[3] - points[0]);
	if (LengthSquared(dx) == 0) {
		Vector3f dy;
		CoordinateSystem(ray.d, &dx, &dy);
	}
	Transform rayFromRender = LookAt(ray.o, ray.o + ray.d, dx);
	pstd::array<Point3f, 16> cp;
	for (size_t i = 0; i < 16; i++) cp[i] = rayFromRender(points[i]);

	// Test ray against bound of projected control points
	Bounds3f patchBounds = Bounds3f(cp[0], cp[1]);
	for (size_t i = 2; i < 16; i++) patchBounds = Union(patchBounds, cp[i]);
	Bounds3f rayBounds(Point3f(0, 0, 0), Point3f(0, 0, Length(ray.d) * tMax));
	if (!Overlaps(rayBounds, patchBounds))
		return false;
	
	return GreedyIntersect(ray, tMax, cp, si);
}

bool BicubicPatch::GreedyIntersect(const Ray &ray, Float tMax, pstd::array<Point3f, 16> const & cp,
                                   pstd::optional<ShapeIntersection> *si) const {
	// RangedPatch definition
	struct RangedPatch {
		Point2f uvMin, uvMax; // Local coordinates for the patch, different from texture coordinates
		Float tLower;
		RangedPatch(const Point2f &uvMin, const Point2f &uvMax, Float tLower)
			: uvMin(uvMin), uvMax(uvMax), tLower(tLower) {}
		bool operator<(const RangedPatch &o) const { return tLower > o.tLower; }
	};
	// Initialize the heap
	std::priority_queue<RangedPatch> heap;
	heap.emplace(Point2f(0, 0), Point2f(1, 1), 0);
	// Initialize variables
	Float tOpt = Infinity;
	
	// Greedily search intersection points
	while (heap.top().tLower < tOpt) {
	}
	return false;
}

}
