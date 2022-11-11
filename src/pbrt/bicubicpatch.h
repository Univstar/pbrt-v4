// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0
//
// This file is maintained by PKU-VCL Lab.

#ifndef PBRT_BICUBICPATCH_H
#define PBRT_BICUBICPATCH_H

#include <pbrt/pbrt.h>

#include <pbrt/base/shape.h>
#include <pbrt/interaction.h>
#include <pbrt/ray.h>
#include <pbrt/util/buffercache.h>
#include <pbrt/util/mesh.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/transform.h>
#include <pbrt/util/vecmath.h>

#include <algorithm>
#include <map>
#include <memory>
#include <vector>

namespace pbrt {

// BilinearPatch Declarations
#if defined(PBRT_BUILD_GPU_RENDERER) && defined(__CUDACC__)
extern PBRT_GPU pstd::vector<const BilinearPatchMesh *> *allBilinearMeshesGPU;
#endif

// BilinearIntersection Definition
struct BilinearIntersection {
    Point2f uv;
    Float t;
    std::string ToString() const;
};

// Bilinear Patch Inline Functions
PBRT_CPU_GPU inline pstd::optional<BilinearIntersection> IntersectBilinearPatch(
    const Ray &ray, Float tMax, Point3f p00, Point3f p10, Point3f p01, Point3f p11) {
    // Find quadratic coefficients for distance from ray to $u$ iso-lines
    Float a = Dot(Cross(p10 - p00, p01 - p11), ray.d);
    Float c = Dot(Cross(p00 - ray.o, ray.d), p01 - p00);
    Float b = Dot(Cross(p10 - ray.o, ray.d), p11 - p10) - (a + c);

    // Solve quadratic for bilinear patch $u$ intersection
    Float u1, u2;
    if (!Quadratic(a, b, c, &u1, &u2))
        return {};

    // Find epsilon _eps_ to ensure that candidate $t$ is greater than zero
    Float eps =
        gamma(10) * (MaxComponentValue(Abs(ray.o)) + MaxComponentValue(Abs(ray.d)) +
                     MaxComponentValue(Abs(p00)) + MaxComponentValue(Abs(p10)) +
                     MaxComponentValue(Abs(p01)) + MaxComponentValue(Abs(p11)));

    // Compute $v$ and $t$ for the first $u$ intersection
    Float t = tMax, u, v;
    if (0 <= u1 && u1 <= 1) {
        // Precompute common terms for $v$ and $t$ computation
        Point3f uo = Lerp(u1, p00, p10);
        Vector3f ud = Lerp(u1, p01, p11) - uo;
        Vector3f deltao = uo - ray.o;
        Vector3f perp = Cross(ray.d, ud);
        Float p2 = LengthSquared(perp);

        // Compute matrix determinants for $v$ and $t$ numerators
        Float v1 =
            Determinant(SquareMatrix<3>(deltao.x, ray.d.x, perp.x, deltao.y, ray.d.y,
                                        perp.y, deltao.z, ray.d.z, perp.z));
        Float t1 = Determinant(SquareMatrix<3>(deltao.x, ud.x, perp.x, deltao.y, ud.y,
                                               perp.y, deltao.z, ud.z, perp.z));

        // Set _u_, _v_, and _t_ if intersection is valid
        if (t1 > p2 * eps && 0 <= v1 && v1 <= p2) {
            u = u1;
            v = v1 / p2;
            t = t1 / p2;
        }
    }

    // Compute $v$ and $t$ for the second $u$ intersection
    if (0 <= u2 && u2 <= 1 && u2 != u1) {
        Point3f uo = Lerp(u2, p00, p10);
        Vector3f ud = Lerp(u2, p01, p11) - uo;
        Vector3f deltao = uo - ray.o;
        Vector3f perp = Cross(ray.d, ud);
        Float p2 = LengthSquared(perp);
        Float v2 =
            Determinant(SquareMatrix<3>(deltao.x, ray.d.x, perp.x, deltao.y, ray.d.y,
                                        perp.y, deltao.z, ray.d.z, perp.z));
        Float t2 = Determinant(SquareMatrix<3>(deltao.x, ud.x, perp.x, deltao.y, ud.y,
                                               perp.y, deltao.z, ud.z, perp.z));
        t2 /= p2;
        if (0 <= v2 && v2 <= p2 && t > t2 && t2 > eps) {
            t = t2;
            u = u2;
            v = v2 / p2;
        }
    }

    // TODO: reject hits with sufficiently small t that we're not sure.
    // Check intersection $t$ against _tMax_ and possibly return intersection
    if (t >= tMax)
        return {};
    return BilinearIntersection{{u, v}, t};
}

// BilinearPatch Definition
class BilinearPatch {
  public:
    // BilinearPatch Public Methods
    BilinearPatch(const BilinearPatchMesh *mesh, int meshIndex, int blpIndex);

    static void Init(Allocator alloc);

    static BilinearPatchMesh *CreateMesh(const Transform *renderFromObject,
                                         bool reverseOrientation,
                                         const ParameterDictionary &parameters,
                                         const FileLoc *loc, Allocator alloc);

    static pstd::vector<Shape> CreatePatches(const BilinearPatchMesh *mesh,
                                             Allocator alloc);

    PBRT_CPU_GPU
    Bounds3f Bounds() const;

    PBRT_CPU_GPU
    pstd::optional<ShapeIntersection> Intersect(const Ray &ray,
                                                Float tMax = Infinity) const;

    PBRT_CPU_GPU
    bool IntersectP(const Ray &ray, Float tMax = Infinity) const;

    PBRT_CPU_GPU
    pstd::optional<ShapeSample> Sample(const ShapeSampleContext &ctx, Point2f u) const;

    PBRT_CPU_GPU
    Float PDF(const ShapeSampleContext &ctx, Vector3f wi) const;

    PBRT_CPU_GPU
    pstd::optional<ShapeSample> Sample(Point2f u) const;

    PBRT_CPU_GPU
    Float PDF(const Interaction &) const;

    PBRT_CPU_GPU
    DirectionCone NormalBounds() const;

    std::string ToString() const;

    PBRT_CPU_GPU
    Float Area() const { return area; }

    PBRT_CPU_GPU
    static SurfaceInteraction InteractionFromIntersection(const BilinearPatchMesh *mesh,
                                                          int blpIndex, Point2f uv,
                                                          Float time, Vector3f wo) {
        // Compute bilinear patch point $\pt{}$, $\dpdu$, and $\dpdv$ for $(u,v)$
        // Get bilinear patch vertices in _p00_, _p01_, _p10_, and _p11_
        const int *v = &mesh->vertexIndices[4 * blpIndex];
        Point3f p00 = mesh->p[v[0]], p10 = mesh->p[v[1]];
        Point3f p01 = mesh->p[v[2]], p11 = mesh->p[v[3]];

        Point3f p = Lerp(uv[0], Lerp(uv[1], p00, p01), Lerp(uv[1], p10, p11));
        Vector3f dpdu = Lerp(uv[1], p10, p11) - Lerp(uv[1], p00, p01);
        Vector3f dpdv = Lerp(uv[0], p01, p11) - Lerp(uv[0], p00, p10);

        // Compute $(s,t)$ texture coordinates at bilinear patch $(u,v)$
        Point2f st = uv;
        Float duds = 1, dudt = 0, dvds = 0, dvdt = 1;
        if (mesh->uv) {
            // Compute texture coordinates for bilinear patch intersection point
            Point2f uv00 = mesh->uv[v[0]], uv10 = mesh->uv[v[1]];
            Point2f uv01 = mesh->uv[v[2]], uv11 = mesh->uv[v[3]];
            st = Lerp(uv[0], Lerp(uv[1], uv00, uv01), Lerp(uv[1], uv10, uv11));

            // Update bilinear patch $\dpdu$ and $\dpdv$ accounting for $(s,t)$
            // Compute partial derivatives of $(u,v)$ with respect to $(s,t)$
            Vector2f dstdu = Lerp(uv[1], uv10, uv11) - Lerp(uv[1], uv00, uv01);
            Vector2f dstdv = Lerp(uv[0], uv01, uv11) - Lerp(uv[0], uv00, uv10);
            duds = std::abs(dstdu[0]) < 1e-8f ? 0 : 1 / dstdu[0];
            dvds = std::abs(dstdv[0]) < 1e-8f ? 0 : 1 / dstdv[0];
            dudt = std::abs(dstdu[1]) < 1e-8f ? 0 : 1 / dstdu[1];
            dvdt = std::abs(dstdv[1]) < 1e-8f ? 0 : 1 / dstdv[1];

            // Compute partial derivatives of $\pt{}$ with respect to $(s,t)$
            Vector3f dpds = dpdu * duds + dpdv * dvds;
            Vector3f dpdt = dpdu * dudt + dpdv * dvdt;

            // Set _dpdu_ and _dpdv_ to updated partial derivatives
            if (Cross(dpds, dpdt) != Vector3f(0, 0, 0)) {
                if (Dot(Cross(dpdu, dpdv), Cross(dpds, dpdt)) < 0)
                    dpdt = -dpdt;
                DCHECK_GE(Dot(Normalize(Cross(dpdu, dpdv)), Normalize(Cross(dpds, dpdt))),
                          -1e-3);
                dpdu = dpds;
                dpdv = dpdt;
            }
        }

        // Find partial derivatives $\dndu$ and $\dndv$ for bilinear patch
        Vector3f d2Pduu(0, 0, 0), d2Pdvv(0, 0, 0);
        Vector3f d2Pduv = (p00 - p01) + (p11 - p10);
        // Compute coefficients for fundamental forms
        Float E = Dot(dpdu, dpdu), F = Dot(dpdu, dpdv), G = Dot(dpdv, dpdv);
        Vector3f n = Normalize(Cross(dpdu, dpdv));
        Float e = Dot(n, d2Pduu), f = Dot(n, d2Pduv), g = Dot(n, d2Pdvv);

        // Compute $\dndu$ and $\dndv$ from fundamental form coefficients
        Float EGF2 = DifferenceOfProducts(E, G, F, F);
        Float invEGF2 = (EGF2 == 0) ? Float(0) : 1 / EGF2;
        Normal3f dndu =
            Normal3f((f * F - e * G) * invEGF2 * dpdu + (e * F - f * E) * invEGF2 * dpdv);
        Normal3f dndv =
            Normal3f((g * F - f * G) * invEGF2 * dpdu + (f * F - g * E) * invEGF2 * dpdv);

        // Update $\dndu$ and $\dndv$ to account for $(s,t)$ parameterization
        Normal3f dnds = dndu * duds + dndv * dvds;
        Normal3f dndt = dndu * dudt + dndv * dvdt;
        dndu = dnds;
        dndv = dndt;

        // Initialize bilinear patch intersection point error _pError_
        Point3f pAbsSum = Abs(p00) + Abs(p01) + Abs(p10) + Abs(p11);
        Vector3f pError = gamma(6) * Vector3f(pAbsSum);

        // Initialize _SurfaceInteraction_ for bilinear patch intersection
        int faceIndex = mesh->faceIndices ? mesh->faceIndices[blpIndex] : 0;
        bool flipNormal = mesh->reverseOrientation ^ mesh->transformSwapsHandedness;
        SurfaceInteraction isect(Point3fi(p, pError), st, wo, dpdu, dpdv, dndu, dndv,
                                 time, flipNormal, faceIndex);

        // Compute bilinear patch shading normal if necessary
        if (mesh->n) {
            // Compute shading normals for bilinear patch intersection point
            Normal3f n00 = mesh->n[v[0]], n10 = mesh->n[v[1]];
            Normal3f n01 = mesh->n[v[2]], n11 = mesh->n[v[3]];
            Normal3f ns = Lerp(uv[0], Lerp(uv[1], n00, n01), Lerp(uv[1], n10, n11));
            if (LengthSquared(ns) > 0) {
                ns = Normalize(ns);
                // Set shading geometry for bilinear patch intersection
                Normal3f dndu = Lerp(uv[1], n10, n11) - Lerp(uv[1], n00, n01);
                Normal3f dndv = Lerp(uv[0], n01, n11) - Lerp(uv[0], n00, n10);
                // Update $\dndu$ and $\dndv$ to account for $(s,t)$ parameterization
                Normal3f dnds = dndu * duds + dndv * dvds;
                Normal3f dndt = dndu * dudt + dndv * dvdt;
                dndu = dnds;
                dndv = dndt;

                Transform r = RotateFromTo(Vector3f(Normalize(isect.n)), Vector3f(ns));
                isect.SetShadingGeometry(ns, r(dpdu), r(dpdv), dndu, dndv, true);
            }
        }

        return isect;
    }

  private:
    // BilinearPatch Private Methods
    PBRT_CPU_GPU
    const BilinearPatchMesh *GetMesh() const {
#ifdef PBRT_IS_GPU_CODE
        return (*allBilinearMeshesGPU)[meshIndex];
#else
        return (*allMeshes)[meshIndex];
#endif
    }

    PBRT_CPU_GPU
    bool IsRectangle(const BilinearPatchMesh *mesh) const {
        // Get bilinear patch vertices in _p00_, _p01_, _p10_, and _p11_
        const int *v = &mesh->vertexIndices[4 * blpIndex];
        Point3f p00 = mesh->p[v[0]], p10 = mesh->p[v[1]];
        Point3f p01 = mesh->p[v[2]], p11 = mesh->p[v[3]];

        if (p00 == p01 || p01 == p11 || p11 == p10 || p10 == p00)
            return false;
        // Check if bilinear patch vertices are coplanar
        Normal3f n(Normalize(Cross(p10 - p00, p01 - p00)));
        if (AbsDot(Normalize(p11 - p00), n) > 1e-5f)
            return false;

        // Check if planar vertices form a rectangle
        Point3f pCenter = (p00 + p01 + p10 + p11) / 4;
        Float d2[4] = {DistanceSquared(p00, pCenter), DistanceSquared(p01, pCenter),
                       DistanceSquared(p10, pCenter), DistanceSquared(p11, pCenter)};
        for (int i = 1; i < 4; ++i)
            if (std::abs(d2[i] - d2[0]) / d2[0] > 1e-4f)
                return false;
        return true;
    }

    // BilinearPatch Private Members
    int meshIndex, blpIndex;
    static pstd::vector<const BilinearPatchMesh *> *allMeshes;
    Float area;
    static constexpr Float MinSphericalSampleArea = 1e-4;
};

}

#endif
