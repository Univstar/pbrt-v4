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

// Bicubic Bezier Functions

PBRT_CPU_GPU static Point3f BlossomBezierCurve(pstd::span<const Point3f> cp, Float u0, Float u1, Float u2) {
    Point3f a[3] = {Lerp(u0, cp[0], cp[1]), Lerp(u0, cp[1], cp[2]), Lerp(u0, cp[2], cp[3])};
    Point3f b[2] = {Lerp(u1, a[0], a[1]), Lerp(u1, a[1], a[2])};
    return Lerp(u2, b[0], b[1]);
}

PBRT_CPU_GPU static Point3f BlossomBicubicBezier(pstd::span<const Point3f> cp, Point2f uv0, Point2f uv1, Point2f uv2) {
    Point3f q[4];
    for (int i = 0; i < 4; ++i) {
        q[i] = BlossomBezierCurve(cp.subspan(i * 4, 4), uv0.y, uv1.y, uv2.y);
    }
    return BlossomBezierCurve(q, uv0.x, uv1.x, uv2.x);
}

PBRT_CPU_GPU static Point3f EvaluateBicubicBezier(pstd::span<const Point3f> cp, Point2f uv, Vector3f *dpdu = nullptr, Vector3f *dpdv = nullptr) {
    if (dpdu && dpdv) {
        Point3f cp_u[4], cp_v[4];
        for (int i = 0; i < 4; ++i) {
            cp_u[i] = BlossomBezierCurve(cp.subspan(i * 4, 4), uv.y, uv.y, uv.y);
            Point3f a[3] = {Lerp(uv.x, cp[i], cp[i + 4]),
                            Lerp(uv.x, cp[i + 4], cp[i + 8]),
                            Lerp(uv.x, cp[i + 8], cp[i + 12])};
            Point3f b[2] = {Lerp(uv.x, a[0], a[1]), Lerp(uv.x, a[1], a[2])};
            cp_v[i] = Lerp(uv.x, b[0], b[1]);
        }
        Point3f cp_u1[3] = {Lerp(uv.x, cp_u[0], cp_u[1]), Lerp(uv.x, cp_u[1], cp_u[2]),
                            Lerp(uv.x, cp_u[2], cp_u[3])};
        Point3f cp_u2[2] = {Lerp(uv.x, cp_u1[0], cp_u1[1]),
                            Lerp(uv.x, cp_u1[1], cp_u1[2])};
        if (LengthSquared(cp_u2[1] - cp_u2[0]) > 0)
            *dpdu = 3 * (cp_u2[1] - cp_u2[0]);
        else {
            *dpdu = cp_u[3] - cp_u[0];
        }
        Point3f cp_v1[3] = {Lerp(uv.y, cp_v[0], cp_v[1]), Lerp(uv.y, cp_v[1], cp_v[2]),
                            Lerp(uv.y, cp_v[2], cp_v[3])};
        Point3f cp_v2[2] = {Lerp(uv.y, cp_v1[0], cp_v1[1]),
                            Lerp(uv.y, cp_v1[1], cp_v1[2])};
        if (LengthSquared(cp_v2[1] - cp_v2[0]) > 0)
            *dpdv = 3 * (cp_v2[1] - cp_v2[0]);
        else {
            *dpdv = cp_v[3] - cp_v[0];
        }
        return BlossomBezierCurve(cp_u, uv.x, uv.x, uv.x);
    } else {
        return BlossomBicubicBezier(cp, uv, uv, uv);
    }
}

// Patch Functions

PBRT_CPU_GPU static pstd::array<Point3f, 16> DivideBezierPatch(pstd::span<const Point3f> cp,
                                                               const Bounds2f &uvB) {
    pstd::array<Point3f, 16> divCp;
    // clang-format off
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
    // clang-format on
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
      uvRect(uvRect) {
    for (size_t i = 0; i < 16; ++i)
        this->cp[i] = (*renderFromObject)(cp[i]);
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

SurfaceInteraction BezierPatch::InteractionFromIntersection(const Ray &ray, Bounds2f uvB) const {
    auto pBounds = GetBounds(DivideBezierPatch(cp, uvB));
    return InteractionFromIntersection(ray, (uvB.pMin + uvB.pMax) / 2, pBounds.Diagonal());
}

SurfaceInteraction BezierPatch::InteractionFromIntersection(const Ray &ray,
                                                            Point2f uv, Vector3f pError) const {
    const Vector2f delta = uvRect.Diagonal();
    Vector3f dpdu;
    Vector3f dpdv;
    const Point3f p = EvaluateBicubicBezier(cp, uv, &dpdu, &dpdv);
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
    
    return GreedyIntersect(ray, tMax, cpRay, si);
}

bool BezierPatch::GreedyIntersect(const Ray &ray, Float tMax, pstd::span<const Point3f> cpRay,
                                  pstd::optional<ShapeIntersection> *si) const {
    constexpr Float threshold = 1e-4f;
    // Initialize the heap
    std::priority_queue<DividedPatch> heap;
    heap.emplace(Bounds2f(Point2f(0, 0), Point2f(1, 1)));
    // Initialize variables
    const Float rayLength = Length(ray.d) * tMax;
    
    // Greedily search intersection points
    while (!heap.empty()) {
        const auto cur = heap.top();
        heap.pop();

        // Decide whether the algorithm converges
        if (MaxComponentValue(cur.uvB.Diagonal()) < threshold) {
            if (si != nullptr) {
                const Float tHit = cur.zLower / Length(ray.d);
                const auto intr = InteractionFromIntersection(ray, cur.uvB);
                *si = ShapeIntersection{intr, tHit};
            }
#ifndef PBRT_IS_GPU_CODE
            ++nBezierPatchHits;
#endif
            return true;
        }
        // Set uv of the middle point
        Point2f uvMid = (cur.uvB.pMin + cur.uvB.pMax) / 2;

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

}
