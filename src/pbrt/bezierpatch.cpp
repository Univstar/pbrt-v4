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

// DividedPatch definition
struct DividedPatch {
    // Bezier coordinates bounds for the divided patch, different from texture coordinates
    Bounds2f uvB;

    Float zLower;

    DividedPatch(const Bounds2f &uvB, Float zLower = 0)
        : uvB(uvB), zLower(zLower) {}
    
    bool operator<(const DividedPatch &o) const { return zLower > o.zLower; }
};

static auto DivideBezierPatch(const pstd::array<Point3f, 16> &cp,
                              const Bounds2f &uvB) {
    pstd::array<Point3f, 16> divCp;
    // TODO
    return std::pair(DividedPatch(uvB), std::move(divCp));
}

static Bounds3f GetBounds(const pstd::array<Point3f, 16> &cp) {
    Bounds3f bounds = Bounds3f(cp[0], cp[1]);
    for (size_t i = 2; i < 16; ++i) bounds = Union(bounds, cp[i]);
    return bounds;
}

static bool OverlapsRay(const Bounds3f &bounds, Float rayLength) {
    bool x = bounds.pMin.x <= 0 && 0 <= bounds.pMax.x;
    bool y = bounds.pMin.y <= 0 && 0 <= bounds.pMax.y;
    bool z = bounds.pMin.z <= rayLength && 0 <= bounds.pMax.z;
    return x && y && z;
}

static std::tuple<bool, Float, Float> NewtonSearch(const pstd::array<Point3f, 16> &cp, Point2f &uv) {
    return {};
}

STAT_PERCENT("Intersections/Ray-bezier patch intersection tests", nBezierPatchHits, nBezierPatchTests);
STAT_COUNTER("Geometry/BezierPatches", nBezierPatches);

BezierPatch *BezierPatch::Create(const Transform *renderFromObject,
                                 bool reverseOrientation,
                                 const ParameterDictionary &parameters,
                                 const FileLoc *loc, Allocator alloc) {
    std::vector<Point3f> P = parameters.GetPoint3fArray("P");
    std::vector<Point2f> uvs = parameters.GetPoint2fArray("uv");

    if (P.size() != 16) {
        Error(loc, "Number of vertex positions \"P\" for bezier patch must be 16.");
        return {};
    }
    if (uvs.empty()) {
        uvs = {Point2f(0, 0), Point2f(1, 1)};
    } else if (uvs.size() != 2) {
        Error(loc, "Number of \"uv\"s for bezier patch must be 2. "
                   "Discarding uvs.");
        uvs = {Point2f(0, 0), Point2f(1, 1)};
    }

    pstd::array<Point3f, 16> cp;
    for (size_t i = 0; i < 16; ++i) cp[i] = P[i];

    return alloc.new_object<BezierPatch>(renderFromObject, reverseOrientation, cp, uvs[0], uvs[1]);
}

std::string BezierPatch::ToString() const {
    return StringPrintf("[ BezierPatch reverseOrientation: %s transformSwapsHandedness: %s "
                        "cp[0]: %s cp[4]: %s "
                        "cp[8]: %s cp[12]: %s",
                        reverseOrientation, transformSwapsHandedness,
                        cp[0], cp[4], cp[8], cp[12]);
}

BezierPatch::BezierPatch(const Transform *renderFromObject, bool reverseOrientation,
                         const pstd::array<Point3f, 16> &cp,
                         const Point2f &uvMin, const Point2f &uvMax)
    : reverseOrientation(reverseOrientation),
      transformSwapsHandedness(renderFromObject->SwapsHandedness()),
      cp(cp),
      uvMin(uvMin),
      uvMax(uvMax) {
    for (Point3f &pt : this->cp)
        pt = (*renderFromObject)(pt);
    ++nBezierPatches;
}

Bounds3f BezierPatch::Bounds() const {
    return GetBounds(cp);
}

DirectionCone BezierPatch::NormalBounds() const {
    return DirectionCone::EntireSphere();
}

pstd::optional<ShapeIntersection> BezierPatch::Intersect(const Ray &ray, Float tMax) const {
    pstd::optional<ShapeIntersection> si;
    IntersectRay(ray, tMax, &si);
    return si;
}

bool BezierPatch::IntersectP(const Ray &ray, Float tMax) const {
    return IntersectRay(ray, tMax, nullptr);
}

Float BezierPatch::Area() const {
    // TODO
    return 0;
}

pstd::optional<ShapeSample> BezierPatch::Sample(Point2f u) const {
    LOG_FATAL("BezierPatch::Sample not implemented.");
    return {};
}

Float BezierPatch::PDF(const Interaction &) const {
    LOG_FATAL("BezierPatch::PDF not implemented.");
    return {};
}

pstd::optional<ShapeSample> BezierPatch::Sample(const ShapeSampleContext &ctx,
                                         Point2f u) const {
    LOG_FATAL("BezierPatch::Sample not implemented.");
    return {};
}

Float BezierPatch::PDF(const ShapeSampleContext &ctx, Vector3f wi) const {
    LOG_FATAL("BezierPatch::PDF not implemented.");
    return {};
}

bool BezierPatch::IntersectRay(const Ray &ray, Float tMax,
                               pstd::optional<ShapeIntersection> *si) const {
#ifndef PBRT_IS_GPU_CODE
    ++nBezierPatchTests;
#endif
    // Project bezier patch control points to plane perpendicular to ray
    Vector3f dx = Cross(ray.d, cp[3] - cp[0]);
    if (LengthSquared(dx) == 0) {
        Vector3f dy;
        CoordinateSystem(ray.d, &dx, &dy);
    }
    Transform rayFromRender = LookAt(ray.o, ray.o + ray.d, dx);
    pstd::array<Point3f, 16> cpRay;
    for (size_t i = 0; i < 16; ++i) cpRay[i] = rayFromRender(cp[i]);

    // Test ray against bound of projected control points
    Bounds3f bounds = GetBounds(cpRay);
    if (!OverlapsRay(bounds, Length(ray.d) * tMax))
        return false;
    
    return GreedyIntersect(ray, tMax, cpRay, si);
}

bool BezierPatch::GreedyIntersect(const Ray &ray, Float tMax, pstd::array<Point3f, 16> const & cpRay,
                                  pstd::optional<ShapeIntersection> *si) const {
    // Initialize the heap
    std::priority_queue<DividedPatch> heap;
    heap.emplace(Bounds2f(Point2f(0, 0), Point2f(1, 1)));
    // Initialize variables
    const Float rayLength = Length(ray.d) * tMax;
    Float zOpt = Infinity;
    Point2f optUv(-1, -1);
    Float optErr = -1;
    
    // Greedily search intersection points
    while (heap.top().zLower < zOpt) {
        auto const cur = heap.top();
        heap.pop();
        // TODO: Decide whether the algorithm converges        
        Point2f uvMid = (cur.uvB.pMin + cur.uvB.pMax) / 2;
        Bounds2f uvCutB;
        auto [converged, zTent, err] = NewtonSearch(cpRay, uvMid);
        if (converged && Inside(uvMid, cur.uvB)) {
            if (si == nullptr) return true;
            else if (zTent < zOpt) {
                zOpt = zTent;
                optUv = uvMid;
                optErr = err;
                Vector2f delta = cur.uvB.Diagonal() / 10;
                uvCutB = Union(Bounds2f(uvMid - delta, uvMid + delta), cur.uvB);
            }
        } else {
            uvMid = (cur.uvB.pMin + cur.uvB.pMax) / 2;
            uvCutB = Bounds2f(uvMid);
        }

        // Divide the current patch into four pieces
        for (size_t i = 0; i < 4; ++i) {
            Bounds2f divUvB(cur.uvB.Corner(i), uvCutB.Corner((i + i + (i < 2)) % 4));
            if (divUvB.IsDegenerate()) continue;

            auto [divPatch, divCpRay] = DivideBezierPatch(cpRay, divUvB);

            Bounds3f bounds = GetBounds(divCpRay);
            if (!OverlapsRay(bounds, rayLength)) continue;

            divPatch.zLower = bounds.pMin.z;
            heap.push(divPatch);
        }
    }

    return false;
}

}
