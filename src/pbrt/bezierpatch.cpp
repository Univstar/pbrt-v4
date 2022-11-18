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

static constexpr Float MinDeltaUV = 1e-5f;

static constexpr size_t MaxNewtonIterations = 5;
static constexpr Float NewtonConvergenceThreshold = 1e-6f;

static constexpr size_t CuttingNeighbor[4] = { 1, 3, 0, 2 };
static constexpr Float CuttingRatioPositive = .2f;
static constexpr Float CuttingRatioNegative = .05f;

// DividedPatch definition
struct DividedPatch {
    // Bezier coordinates bounds for the divided patch, different from texture coordinates
    Bounds2f uvB;

    Float zLower;

    DividedPatch(const Bounds2f &uvB, Float zLower = 0)                       
        : uvB(uvB), zLower(zLower) {}
    
    bool operator<(const DividedPatch &o) const { return zLower > o.zLower; }
};

// Bicubic Bezier Functions

PBRT_CPU_GPU static Point3f BlossomBicubicBezier(pstd::span<const Point3f> cp, Point2f uv0, Point2f uv1, Point2f uv2) {
    pstd::array<Point3f, 4> q;
    for (size_t i = 0; i < 4; ++i) {
        q[i] = BlossomCubicBezier(cp.subspan(i * 4, 4), uv0.y, uv1.y, uv2.y);
    }
    return BlossomCubicBezier<Point3f>(q, uv0.x, uv1.x, uv2.x);
}

PBRT_CPU_GPU static Point3f EvaluateBicubicBezier(pstd::span<const Point3f> cp, Point2f uv, Vector3f *dpdu = nullptr, Vector3f *dpdv = nullptr) {
    if (dpdu && dpdv) {
        pstd::array<Point3f, 4> cpU, cpV;
        for (size_t i = 0; i < 4; ++i) {
            cpU[i] = BlossomCubicBezier(cp.subspan(i * 4, 4), uv.y, uv.y, uv.y);
            pstd::array<Point3f, 3> a = {Lerp(uv.x, cp[i], cp[i + 4]),
                                         Lerp(uv.x, cp[i + 4], cp[i + 8]),
                                         Lerp(uv.x, cp[i + 8], cp[i + 12])};
            pstd::array<Point3f, 2> b = {Lerp(uv.x, a[0], a[1]),
                                         Lerp(uv.x, a[1], a[2])};
            cpV[i] = Lerp(uv.x, b[0], b[1]);
        }
        pstd::array<Point3f, 3> cpU1 = {Lerp(uv.x, cpU[0], cpU[1]),
                                        Lerp(uv.x, cpU[1], cpU[2]),
                                        Lerp(uv.x, cpU[2], cpU[3])};
        pstd::array<Point3f, 2> cpU2 = {Lerp(uv.x, cpU1[0], cpU1[1]),
                                        Lerp(uv.x, cpU1[1], cpU1[2])};
        if (LengthSquared(cpU2[1] - cpU2[0]) > 0)
            *dpdu = 3 * (cpU2[1] - cpU2[0]);
        else {
            *dpdu = cpU[3] - cpU[0];
        }
        pstd::array<Point3f, 3> cpV1 = {Lerp(uv.y, cpV[0], cpV[1]),
                                        Lerp(uv.y, cpV[1], cpV[2]),
                                        Lerp(uv.y, cpV[2], cpV[3])};
        pstd::array<Point3f, 2> cpV2 = {Lerp(uv.y, cpV1[0], cpV1[1]),
                                        Lerp(uv.y, cpV1[1], cpV1[2])};
        if (LengthSquared(cpV2[1] - cpV2[0]) > 0)
            *dpdv = 3 * (cpV2[1] - cpV2[0]);
        else {
            *dpdv = cpV[3] - cpV[0];
        }
        return BlossomCubicBezier<Point3f>(cpU, uv.x, uv.x, uv.x);
    } else {
        return BlossomBicubicBezier(cp, uv, uv, uv);
    }
}

// Patch Functions

PBRT_CPU_GPU static pstd::array<Point3f, 16> DivideBezierPatch(pstd::span<const Point3f> cp,
                                                               const Bounds2f &uvB) {
    pstd::array<Point3f, 16> divCp;
    divCp[0]  = BlossomBicubicBezier(cp, uvB.Corner(0), uvB.Corner(0), uvB.Corner(0));
    divCp[1]  = BlossomBicubicBezier(cp, uvB.Corner(0), uvB.Corner(0), uvB.Corner(2));
    divCp[2]  = BlossomBicubicBezier(cp, uvB.Corner(0), uvB.Corner(2), uvB.Corner(2));
    divCp[3]  = BlossomBicubicBezier(cp, uvB.Corner(2), uvB.Corner(2), uvB.Corner(2));
    divCp[4]  = BlossomBicubicBezier(cp, uvB.Corner(0), uvB.Corner(0), uvB.Corner(1));
    divCp[5]  = BlossomBicubicBezier(cp, uvB.Corner(0), uvB.Corner(0), uvB.Corner(3));
    divCp[6]  = BlossomBicubicBezier(cp, uvB.Corner(0), uvB.Corner(2), uvB.Corner(3));
    divCp[7]  = BlossomBicubicBezier(cp, uvB.Corner(2), uvB.Corner(2), uvB.Corner(3));
    divCp[8]  = BlossomBicubicBezier(cp, uvB.Corner(0), uvB.Corner(1), uvB.Corner(1));
    divCp[9]  = BlossomBicubicBezier(cp, uvB.Corner(0), uvB.Corner(1), uvB.Corner(3));
    divCp[10] = BlossomBicubicBezier(cp, uvB.Corner(0), uvB.Corner(3), uvB.Corner(3));
    divCp[11] = BlossomBicubicBezier(cp, uvB.Corner(2), uvB.Corner(3), uvB.Corner(3));
    divCp[12] = BlossomBicubicBezier(cp, uvB.Corner(1), uvB.Corner(1), uvB.Corner(1));
    divCp[13] = BlossomBicubicBezier(cp, uvB.Corner(1), uvB.Corner(1), uvB.Corner(3));
    divCp[14] = BlossomBicubicBezier(cp, uvB.Corner(1), uvB.Corner(3), uvB.Corner(3));
    divCp[15] = BlossomBicubicBezier(cp, uvB.Corner(3), uvB.Corner(3), uvB.Corner(3));
    return divCp;
}

PBRT_CPU_GPU static Bounds3f GetBounds(const pstd::span<const Point3f> cp) {
    Bounds3f bounds = Bounds3f(cp[0], cp[1]);
    for (size_t i = 2; i < 16; ++i) bounds = Union(bounds, cp[i]);
    return bounds;
}

PBRT_CPU_GPU static bool OverlapsRay(const Bounds3f &bounds, Float rayLength) {
    bool x = bounds.pMin.x <= 0 && 0 <= bounds.pMax.x;
    bool y = bounds.pMin.y <= 0 && 0 <= bounds.pMax.y;
    bool z = bounds.pMin.z <= rayLength && 0 <= bounds.pMax.z;
    return x && y && z;
}

static Float NewtonSearch(const pstd::span<const Point3f> cpRay, Point2f &uv) {
    for (size_t i = 0; i < MaxNewtonIterations; ++i) {
        Vector3f dpdu;
        Vector3f dpdv;
        const Point3f p = EvaluateBicubicBezier(cpRay, uv, &dpdu, &dpdv);
        const Float determinant = dpdu.x * dpdv.y - dpdv.x * dpdu.y;
        const Vector2f delta = Vector2f(dpdv.x * p.y - dpdv.y * p.x, dpdu.y * p.x - dpdu.x * p.y) / determinant;
        uv += delta;
    }
    const Point3f p = EvaluateBicubicBezier(cpRay, uv);
    if (p.x * p.x + p.y * p.y < NewtonConvergenceThreshold * NewtonConvergenceThreshold) {
        return p.z;
    } else {
        return Infinity;
    }
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

    return alloc.new_object<BezierPatch>(renderFromObject, reverseOrientation, P, Bounds2f(uvs[0], uvs[1]));
}

std::string BezierPatch::ToString() const {
    return StringPrintf("[ BezierPatch reverseOrientation: %s transformSwapsHandedness: %s "
                        "cp[0]: %s cp[4]: %s "
                        "cp[8]: %s cp[12]: %s",
                        reverseOrientation, transformSwapsHandedness,
                        cp[0], cp[4], cp[8], cp[12]);
}

BezierPatch::BezierPatch(const Transform *renderFromObject, bool reverseOrientation,
                         const pstd::span<const Point3f> cp,
                         const Bounds2f &uvRect)
    : reverseOrientation(reverseOrientation),
      transformSwapsHandedness(renderFromObject->SwapsHandedness()),
      uvRect(uvRect),
      cpAbsSum(Point3f(0, 0, 0)) {
    for (size_t i = 0; i < 16; ++i) {
        this->cp[i] = (*renderFromObject)(cp[i]);
        cpAbsSum += Abs(this->cp[i]);
    }
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

SurfaceInteraction BezierPatch::InteractionFromIntersection(const Ray &ray,
                                                            Point2f uv) const {
    const Vector2f delta = uvRect.Diagonal();
    Vector3f dpdu;
    Vector3f dpdv;
    const Point3f p = EvaluateBicubicBezier(cp, uv, &dpdu, &dpdv);
    const Vector3f pError = gamma(18) * Vector3f(cpAbsSum);
    dpdu /= delta.x, dpdv /= delta.y;
    Normal3f dndu(0, 0, 0);
    Normal3f dndv(0, 0, 0);
    bool flipNormal = reverseOrientation ^ transformSwapsHandedness;
    return SurfaceInteraction(Point3fi(p, pError), uvRect.Lerp(uv), -ray.d, dpdu, dpdv, dndu, dndv, ray.time, flipNormal);
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
    
    return GreedyIntersectNewton(ray, tMax, cpRay, si);
}

bool BezierPatch::GreedyIntersect(const Ray &ray, Float tMax, pstd::span<const Point3f> cpRay,
                                  pstd::optional<ShapeIntersection> *si) const {
    // Initialize the heap
    std::priority_queue<DividedPatch> heap;
    heap.emplace(Bounds2f(Point2f(0, 0), Point2f(1, 1)));
    // Initialize variables
    const Float rayLength = Length(ray.d) * tMax;
    
    // Greedily search intersection points
    while (!heap.empty()) {
        const auto cur = heap.top();
        heap.pop();

        // Set uv of the middle point
        Point2f uvMid = (cur.uvB.pMin + cur.uvB.pMax) / 2;

        // Decide whether the algorithm converges
        if (MaxComponentValue(cur.uvB.Diagonal()) < MinDeltaUV) {
            if (si != nullptr) {
                const Float tHit = cur.zLower / Length(ray.d);
                const auto intr = InteractionFromIntersection(ray, uvMid);
                *si = ShapeIntersection{intr, tHit};
            }
#ifndef PBRT_IS_GPU_CODE
            ++nBezierPatchHits;
#endif
            return true;
        }

        // Divide the current patch into four pieces
        for (size_t i = 0; i < 4; ++i) {
            Bounds2f divUvB(cur.uvB.Corner(i), uvMid);

            auto divCpRay = DivideBezierPatch(cpRay, divUvB);
            // Test ray against bound of divided patch
            Bounds3f bounds = GetBounds(divCpRay);
            if (!OverlapsRay(bounds, rayLength)) continue;
            // Push new patch into heap
            heap.emplace(divUvB, bounds.pMin.z);
        }
    }

    return false;
}

bool BezierPatch::GreedyIntersectNewton(const Ray &ray, Float tMax, pstd::span<const Point3f> cpRay,
                                        pstd::optional<ShapeIntersection> *si) const {
    // Initialize the heap
    std::priority_queue<DividedPatch> heap;
    heap.emplace(Bounds2f(Point2f(0, 0), Point2f(1, 1)));
    // Initialize variables
    const Float rayLength = Length(ray.d) * tMax;
    Float zOpt = Infinity;
    Point2f optUV(-1, -1);
    
    // Greedily search intersection points
    while (!heap.empty()) {
        const auto cur = heap.top();
        // Break if optimized.
        if (cur.zLower >= zOpt) break;
        heap.pop();
        // Break if patch is too small
        if (MinComponentValue(cur.uvB.Diagonal()) < MinDeltaUV) break;

        // Set uv of the middle point
        Point2f uvMid = (cur.uvB.pMin + cur.uvB.pMax) / 2;
        // Set cutoff bounds
        Bounds2f cutUvB(uvMid);

        // Decide whether the Newton algorithm converges
        if (const Float zTent = NewtonSearch(cpRay, uvMid); Inside(uvMid, cur.uvB)) {
            if (0 < zTent && zTent < zOpt) {
                if (si == nullptr) return true;
                zOpt = zTent, optUV = uvMid;
            }
            if (zTent < Infinity) {
                const Vector2f delta = cur.uvB.Diagonal() * (zTent > 0 ? CuttingRatioPositive : CuttingRatioNegative) * .5f;
                cutUvB = pbrt::Intersect(Bounds2f(uvMid - delta, uvMid + delta), cur.uvB);
            }
        }

        // Divide the current patch into four pieces
        for (size_t i = 0; i < 4; ++i) {
            Bounds2f divUvB(cur.uvB.Corner(i), cutUvB.Corner(CuttingNeighbor[i]));
            if (divUvB.IsDegenerate()) continue;

            auto divCpRay = DivideBezierPatch(cpRay, divUvB);
            // Test ray against bound of divided patch
            Bounds3f bounds = GetBounds(divCpRay);
            if (!OverlapsRay(bounds, rayLength)) continue;
            // Push new patch into heap
            heap.emplace(divUvB, bounds.pMin.z);
        }
    }

    if (zOpt < Infinity) {
        const Float tHit = zOpt / Length(ray.d);
        const auto intr = InteractionFromIntersection(ray, optUV);
        *si = ShapeIntersection{intr, tHit};
#ifndef PBRT_IS_GPU_CODE
        ++nBezierPatchHits;
#endif
    }
    return zOpt < Infinity;
}

}
