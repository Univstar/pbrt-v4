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

STAT_PIXEL_RATIO("Intersections/Ray-bilinear patch intersection tests", nBLPHits,
                 nBLPTests);

// BilinearPatch Method Definitions
std::string BilinearIntersection::ToString() const {
    return StringPrintf("[ BilinearIntersection uv: %s t: %f", uv, t);
}

BilinearPatchMesh *BilinearPatch::CreateMesh(const Transform *renderFromObject,
                                             bool reverseOrientation,
                                             const ParameterDictionary &parameters,
                                             const FileLoc *loc, Allocator alloc) {
    std::vector<int> vertexIndices = parameters.GetIntArray("indices");
    std::vector<Point3f> P = parameters.GetPoint3fArray("P");
    std::vector<Point2f> uv = parameters.GetPoint2fArray("uv");

    if (vertexIndices.empty()) {
        if (P.size() == 4)
            // single patch
            vertexIndices = {0, 1, 2, 3};
        else {
            Error(loc, "Vertex indices \"indices\" must be provided with "
                       "bilinear patch mesh shape.");
            return {};
        }
    } else if ((vertexIndices.size() % 4) != 0u) {
        Error(loc,
              "Number of vertex indices %d not a multiple of 4. Discarding %d "
              "excess.",
              int(vertexIndices.size()), int(vertexIndices.size() % 4));
        while ((vertexIndices.size() % 4) != 0u)
            vertexIndices.pop_back();
    }

    if (P.empty()) {
        Error(loc, "Vertex positions \"P\" must be provided with bilinear "
                   "patch mesh shape.");
        return {};
    }

    if (!uv.empty() && uv.size() != P.size()) {
        Error(loc, "Number of \"uv\"s for bilinear patch mesh must match \"P\"s. "
                   "Discarding uvs.");
        uv = {};
    }

    std::vector<Normal3f> N = parameters.GetNormal3fArray("N");
    if (!N.empty() && N.size() != P.size()) {
        Error(loc, "Number of \"N\"s for bilinear patch mesh must match \"P\"s. "
                   "Discarding \"N\"s.");
        N = {};
    }

    for (size_t i = 0; i < vertexIndices.size(); ++i)
        if (vertexIndices[i] >= P.size()) {
            Error(loc,
                  "Bilinear patch mesh has out of-bounds vertex index %d (%d "
                  "\"P\" "
                  "values were given. Discarding this mesh.",
                  vertexIndices[i], (int)P.size());
            return {};
        }

    std::vector<int> faceIndices = parameters.GetIntArray("faceIndices");
    if (!faceIndices.empty() && faceIndices.size() != vertexIndices.size() / 4) {
        Error(loc,
              "Number of face indices %d does not match number of bilinear "
              "patches %d. "
              "Discarding face indices.",
              int(faceIndices.size()), int(vertexIndices.size() / 4));
        faceIndices = {};
    }

    // Grab this before the vertexIndices are std::moved...
    size_t nBlps = vertexIndices.size() / 4;

    std::string filename =
        ResolveFilename(parameters.GetOneString("emissionfilename", ""));
    PiecewiseConstant2D *imageDist = nullptr;
    if (!filename.empty()) {
        if (!uv.empty())
            Error(loc, "\"emissionfilename\" is currently ignored for bilinear patches "
                       "if \"uv\" coordinates have been provided--sorry!");
        else {
            ImageAndMetadata im = Image::Read(filename, alloc);
            // Account for v inversion in DiffuseAreaLight lookup, which in turn is there
            // to match ImageTexture...
            im.image.FlipY();
            Bounds2f domain = Bounds2f(Point2f(0, 0), Point2f(1, 1));
            Array2D<Float> d = im.image.GetSamplingDistribution();
            imageDist = alloc.new_object<PiecewiseConstant2D>(d, domain, alloc);
        }
    }

    return alloc.new_object<BilinearPatchMesh>(
        *renderFromObject, reverseOrientation, std::move(vertexIndices), std::move(P),
        std::move(N), std::move(uv), std::move(faceIndices), imageDist, alloc);
}

pstd::vector<Shape> BilinearPatch::CreatePatches(const BilinearPatchMesh *mesh,
                                                 Allocator alloc) {
    static std::mutex allMeshesLock;
    allMeshesLock.lock();
    CHECK_LT(allMeshes->size(), 1 << 31);
    int meshIndex = int(allMeshes->size());
    allMeshes->push_back(mesh);
    allMeshesLock.unlock();

    pstd::vector<Shape> blps(mesh->nPatches, alloc);
    BilinearPatch *patches = alloc.allocate_object<BilinearPatch>(mesh->nPatches);
    for (int i = 0; i < mesh->nPatches; ++i) {
        alloc.construct(&patches[i], mesh, meshIndex, i);
        blps[i] = &patches[i];
    }

    return blps;
}

pstd::vector<const BilinearPatchMesh *> *BilinearPatch::allMeshes;
#if defined(PBRT_BUILD_GPU_RENDERER)
PBRT_GPU pstd::vector<const BilinearPatchMesh *> *allBilinearMeshesGPU;
#endif

void BilinearPatch::Init(Allocator alloc) {
    allMeshes = alloc.new_object<pstd::vector<const BilinearPatchMesh *>>(alloc);
#if defined(PBRT_BUILD_GPU_RENDERER)
    if (Options->useGPU)
        CUDA_CHECK(
            cudaMemcpyToSymbol(allBilinearMeshesGPU, &allMeshes, sizeof(allMeshes)));
#endif
}

STAT_MEMORY_COUNTER("Memory/Bilinear patches", blpBytes);

// BilinearPatch Method Definitions
BilinearPatch::BilinearPatch(const BilinearPatchMesh *mesh, int meshIndex, int blpIndex)
    : meshIndex(meshIndex), blpIndex(blpIndex) {
    blpBytes += sizeof(*this);
    // Store area of bilinear patch in _area_
    // Get bilinear patch vertices in _p00_, _p01_, _p10_, and _p11_
    const int *v = &mesh->vertexIndices[4 * blpIndex];
    Point3f p00 = mesh->p[v[0]], p10 = mesh->p[v[1]];
    Point3f p01 = mesh->p[v[2]], p11 = mesh->p[v[3]];

    if (IsRectangle(mesh))
        area = Distance(p00, p01) * Distance(p00, p10);
    else {
        // Compute approximate area of bilinear patch
        // FIXME: it would be good to skip this for flat patches, or to
        // be adaptive based on curvature in some manner
        constexpr int na = 3;
        Point3f p[na + 1][na + 1];
        for (int i = 0; i <= na; ++i) {
            Float u = Float(i) / Float(na);
            for (int j = 0; j <= na; ++j) {
                Float v = Float(j) / Float(na);
                p[i][j] = Lerp(u, Lerp(v, p00, p01), Lerp(v, p10, p11));
            }
        }
        area = 0;
        for (int i = 0; i < na; ++i)
            for (int j = 0; j < na; ++j)
                area += 0.5f * Length(Cross(p[i + 1][j + 1] - p[i][j],
                                            p[i + 1][j] - p[i][j + 1]));
    }
}

Bounds3f BilinearPatch::Bounds() const {
    const BilinearPatchMesh *mesh = GetMesh();
    // Get bilinear patch vertices in _p00_, _p01_, _p10_, and _p11_
    const int *v = &mesh->vertexIndices[4 * blpIndex];
    Point3f p00 = mesh->p[v[0]], p10 = mesh->p[v[1]];
    Point3f p01 = mesh->p[v[2]], p11 = mesh->p[v[3]];

    return Union(Bounds3f(p00, p01), Bounds3f(p10, p11));
}

DirectionCone BilinearPatch::NormalBounds() const {
    const BilinearPatchMesh *mesh = GetMesh();
    // Get bilinear patch vertices in _p00_, _p01_, _p10_, and _p11_
    const int *v = &mesh->vertexIndices[4 * blpIndex];
    Point3f p00 = mesh->p[v[0]], p10 = mesh->p[v[1]];
    Point3f p01 = mesh->p[v[2]], p11 = mesh->p[v[3]];

    // If patch is a triangle, return bounds for single surface normal
    if (p00 == p10 || p10 == p11 || p11 == p01 || p01 == p00) {
        Vector3f dpdu = Lerp(0.5f, p10, p11) - Lerp(0.5f, p00, p01);
        Vector3f dpdv = Lerp(0.5f, p01, p11) - Lerp(0.5f, p00, p10);
        Vector3f n = Normalize(Cross(dpdu, dpdv));
        if (mesh->n) {
            Normal3f ns =
                (mesh->n[v[0]] + mesh->n[v[1]] + mesh->n[v[2]] + mesh->n[v[3]]) / 4;
            n = FaceForward(n, ns);
        } else if (mesh->reverseOrientation ^ mesh->transformSwapsHandedness)
            n = -n;
        return DirectionCone(n);
    }

    // Compute bilinear patch normal _n00_ at $(0,0)$
    Vector3f n00 = Normalize(Cross(p10 - p00, p01 - p00));
    if (mesh->n)
        n00 = FaceForward(n00, mesh->n[v[0]]);
    else if (mesh->reverseOrientation ^ mesh->transformSwapsHandedness)
        n00 = -n00;

    // Compute bilinear patch normals _n10_, _n01_, and _n11_
    Vector3f n10 = Normalize(Cross(p11 - p10, p00 - p10));
    Vector3f n01 = Normalize(Cross(p00 - p01, p11 - p01));
    Vector3f n11 = Normalize(Cross(p01 - p11, p10 - p11));
    if (mesh->n) {
        n10 = FaceForward(n10, mesh->n[v[1]]);
        n01 = FaceForward(n01, mesh->n[v[2]]);
        n11 = FaceForward(n11, mesh->n[v[3]]);
    } else if (mesh->reverseOrientation ^ mesh->transformSwapsHandedness) {
        n10 = -n10;
        n01 = -n01;
        n11 = -n11;
    }

    // Compute average normal and return normal bounds for patch
    Vector3f n = Normalize(n00 + n10 + n01 + n11);
    Float cosTheta = std::min({Dot(n, n00), Dot(n, n01), Dot(n, n10), Dot(n, n11)});
    return DirectionCone(n, Clamp(cosTheta, -1, 1));
}

pstd::optional<ShapeIntersection> BilinearPatch::Intersect(const Ray &ray,
                                                           Float tMax) const {
    const BilinearPatchMesh *mesh = GetMesh();
    // Get bilinear patch vertices in _p00_, _p01_, _p10_, and _p11_
    const int *v = &mesh->vertexIndices[4 * blpIndex];
    Point3f p00 = mesh->p[v[0]], p10 = mesh->p[v[1]];
    Point3f p01 = mesh->p[v[2]], p11 = mesh->p[v[3]];

    pstd::optional<BilinearIntersection> blpIsect =
        IntersectBilinearPatch(ray, tMax, p00, p10, p01, p11);
    if (!blpIsect)
        return {};
    SurfaceInteraction intr =
        InteractionFromIntersection(mesh, blpIndex, blpIsect->uv, ray.time, -ray.d);
    return ShapeIntersection{intr, blpIsect->t};
}

bool BilinearPatch::IntersectP(const Ray &ray, Float tMax) const {
    const BilinearPatchMesh *mesh = GetMesh();
    // Get bilinear patch vertices in _p00_, _p01_, _p10_, and _p11_
    const int *v = &mesh->vertexIndices[4 * blpIndex];
    Point3f p00 = mesh->p[v[0]], p10 = mesh->p[v[1]];
    Point3f p01 = mesh->p[v[2]], p11 = mesh->p[v[3]];

    return IntersectBilinearPatch(ray, tMax, p00, p10, p01, p11).has_value();
}

pstd::optional<ShapeSample> BilinearPatch::Sample(Point2f u) const {
    const BilinearPatchMesh *mesh = GetMesh();
    // Get bilinear patch vertices in _p00_, _p01_, _p10_, and _p11_
    const int *v = &mesh->vertexIndices[4 * blpIndex];
    Point3f p00 = mesh->p[v[0]], p10 = mesh->p[v[1]];
    Point3f p01 = mesh->p[v[2]], p11 = mesh->p[v[3]];

    // Sample bilinear patch parametric $(u,v)$ coordinates
    Float pdf = 1;
    Point2f uv;
    if (mesh->imageDistribution)
        uv = mesh->imageDistribution->Sample(u, &pdf);
    else if (!IsRectangle(mesh)) {
        // Sample patch $(u,v)$ with approximate uniform area sampling
        // Initialize _w_ array with differential area at bilinear patch corners
        pstd::array<Float, 4> w = {
            Length(Cross(p10 - p00, p01 - p00)), Length(Cross(p10 - p00, p11 - p10)),
            Length(Cross(p01 - p00, p11 - p01)), Length(Cross(p11 - p10, p11 - p01))};

        uv = SampleBilinear(u, w);
        pdf = BilinearPDF(uv, w);

    } else
        uv = u;

    // Compute bilinear patch geometric quantities at sampled $(u,v)$
    // Compute $\pt{}$, $\dpdu$, and $\dpdv$ for sampled $(u,v)$
    Point3f pu0 = Lerp(uv[1], p00, p01), pu1 = Lerp(uv[1], p10, p11);
    Point3f p = Lerp(uv[0], pu0, pu1);
    Vector3f dpdu = pu1 - pu0;
    Vector3f dpdv = Lerp(uv[0], p01, p11) - Lerp(uv[0], p00, p10);
    if (LengthSquared(dpdu) == 0 || LengthSquared(dpdv) == 0)
        return {};

    Point2f st = uv;
    if (mesh->uv) {
        // Compute texture coordinates for bilinear patch intersection point
        Point2f uv00 = mesh->uv[v[0]], uv10 = mesh->uv[v[1]];
        Point2f uv01 = mesh->uv[v[2]], uv11 = mesh->uv[v[3]];
        st = Lerp(uv[0], Lerp(uv[1], uv00, uv01), Lerp(uv[1], uv10, uv11));
    }
    // Compute surface normal for sampled bilinear patch $(u,v)$
    Normal3f n = Normal3f(Normalize(Cross(dpdu, dpdv)));
    // Flip normal at sampled $(u,v)$ if necessary
    if (mesh->n) {
        Normal3f n00 = mesh->n[v[0]], n10 = mesh->n[v[1]];
        Normal3f n01 = mesh->n[v[2]], n11 = mesh->n[v[3]];
        Normal3f ns = Lerp(uv[0], Lerp(uv[1], n00, n01), Lerp(uv[1], n10, n11));
        n = FaceForward(n, ns);
    } else if (mesh->reverseOrientation ^ mesh->transformSwapsHandedness)
        n = -n;

    // Compute _pError_ for sampled bilinear patch $(u,v)$
    Point3f pAbsSum = Abs(p00) + Abs(p01) + Abs(p10) + Abs(p11);
    Vector3f pError = gamma(6) * Vector3f(pAbsSum);

    // Return _ShapeSample_ for sampled bilinear patch point
    return ShapeSample{Interaction(Point3fi(p, pError), n, st),
                       pdf / Length(Cross(dpdu, dpdv))};
}

Float BilinearPatch::PDF(const Interaction &intr) const {
    const BilinearPatchMesh *mesh = GetMesh();
    // Get bilinear patch vertices in _p00_, _p01_, _p10_, and _p11_
    const int *v = &mesh->vertexIndices[4 * blpIndex];
    Point3f p00 = mesh->p[v[0]], p10 = mesh->p[v[1]];
    Point3f p01 = mesh->p[v[2]], p11 = mesh->p[v[3]];

    // Compute parametric $(u,v)$ of point on bilinear patch
    Point2f uv = intr.uv;
    if (mesh->uv) {
        Point2f uv00 = mesh->uv[v[0]], uv10 = mesh->uv[v[1]];
        Point2f uv01 = mesh->uv[v[2]], uv11 = mesh->uv[v[3]];
        uv = InvertBilinear(uv, {uv00, uv10, uv01, uv11});
    }

    // Compute PDF for sampling the $(u,v)$ coordinates given by _intr.uv_
    Float pdf;
    if (mesh->imageDistribution)
        pdf = mesh->imageDistribution->PDF(uv);
    else if (!IsRectangle(mesh)) {
        // Initialize _w_ array with differential area at bilinear patch corners
        pstd::array<Float, 4> w = {
            Length(Cross(p10 - p00, p01 - p00)), Length(Cross(p10 - p00, p11 - p10)),
            Length(Cross(p01 - p00, p11 - p01)), Length(Cross(p11 - p10, p11 - p01))};

        pdf = BilinearPDF(uv, w);
    } else
        pdf = 1;

    // Find $\dpdu$ and $\dpdv$ at bilinear patch $(u,v)$
    Point3f pu0 = Lerp(uv[1], p00, p01), pu1 = Lerp(uv[1], p10, p11);
    Vector3f dpdu = pu1 - pu0;
    Vector3f dpdv = Lerp(uv[0], p01, p11) - Lerp(uv[0], p00, p10);

    // Return final bilinear patch area sampling PDF
    return pdf / Length(Cross(dpdu, dpdv));
}

pstd::optional<ShapeSample> BilinearPatch::Sample(const ShapeSampleContext &ctx,
                                                  Point2f u) const {
    const BilinearPatchMesh *mesh = GetMesh();
    // Get bilinear patch vertices in _p00_, _p01_, _p10_, and _p11_
    const int *v = &mesh->vertexIndices[4 * blpIndex];
    Point3f p00 = mesh->p[v[0]], p10 = mesh->p[v[1]];
    Point3f p01 = mesh->p[v[2]], p11 = mesh->p[v[3]];

    // Sample bilinear patch with respect to solid angle from reference point
    Vector3f v00 = Normalize(p00 - ctx.p()), v10 = Normalize(p10 - ctx.p());
    Vector3f v01 = Normalize(p01 - ctx.p()), v11 = Normalize(p11 - ctx.p());
    if (!IsRectangle(mesh) || mesh->imageDistribution ||
        SphericalQuadArea(v00, v10, v11, v01) <= MinSphericalSampleArea) {
        // Sample shape by area and compute incident direction _wi_
        pstd::optional<ShapeSample> ss = Sample(u);
        DCHECK(ss.has_value());
        ss->intr.time = ctx.time;
        Vector3f wi = ss->intr.p() - ctx.p();
        if (LengthSquared(wi) == 0)
            return {};
        wi = Normalize(wi);

        // Convert area sampling PDF in _ss_ to solid angle measure
        ss->pdf /= AbsDot(ss->intr.n, -wi) / DistanceSquared(ctx.p(), ss->intr.p());
        if (IsInf(ss->pdf))
            return {};

        return ss;
    }
    // Sample direction to rectangular bilinear patch
    Float pdf = 1;
    // Warp uniform sample _u_ to account for incident $\cos \theta$ factor
    if (ctx.ns != Normal3f(0, 0, 0)) {
        // Compute $\cos \theta$ weights for rectangle seen from reference point
        pstd::array<Float, 4> w =
            pstd::array<Float, 4>{std::max<Float>(0.01, AbsDot(v00, ctx.ns)),
                                  std::max<Float>(0.01, AbsDot(v10, ctx.ns)),
                                  std::max<Float>(0.01, AbsDot(v01, ctx.ns)),
                                  std::max<Float>(0.01, AbsDot(v11, ctx.ns))};

        u = SampleBilinear(u, w);
        pdf *= BilinearPDF(u, w);
    }

    // Sample spherical rectangle at reference point
    Vector3f eu = p10 - p00, ev = p01 - p00;
    Float quadPDF;
    Point3f p = SampleSphericalRectangle(ctx.p(), p00, eu, ev, u, &quadPDF);
    pdf *= quadPDF;

    // Compute $(u,v)$ and surface normal for sampled point on rectangle
    Point2f uv(Dot(p - p00, eu) / DistanceSquared(p10, p00),
               Dot(p - p00, ev) / DistanceSquared(p01, p00));
    Normal3f n = Normal3f(Normalize(Cross(eu, ev)));
    // Flip normal at sampled $(u,v)$ if necessary
    if (mesh->n) {
        Normal3f n00 = mesh->n[v[0]], n10 = mesh->n[v[1]];
        Normal3f n01 = mesh->n[v[2]], n11 = mesh->n[v[3]];
        Normal3f ns = Lerp(uv[0], Lerp(uv[1], n00, n01), Lerp(uv[1], n10, n11));
        n = FaceForward(n, ns);
    } else if (mesh->reverseOrientation ^ mesh->transformSwapsHandedness)
        n = -n;

    // Compute $(s,t)$ texture coordinates for sampled $(u,v)$
    Point2f st = uv;
    if (mesh->uv) {
        // Compute texture coordinates for bilinear patch intersection point
        Point2f uv00 = mesh->uv[v[0]], uv10 = mesh->uv[v[1]];
        Point2f uv01 = mesh->uv[v[2]], uv11 = mesh->uv[v[3]];
        st = Lerp(uv[0], Lerp(uv[1], uv00, uv01), Lerp(uv[1], uv10, uv11));
    }

    return ShapeSample{Interaction(p, n, ctx.time, st), pdf};
}

Float BilinearPatch::PDF(const ShapeSampleContext &ctx, Vector3f wi) const {
    const BilinearPatchMesh *mesh = GetMesh();
    // Get bilinear patch vertices in _p00_, _p01_, _p10_, and _p11_
    const int *v = &mesh->vertexIndices[4 * blpIndex];
    Point3f p00 = mesh->p[v[0]], p10 = mesh->p[v[1]];
    Point3f p01 = mesh->p[v[2]], p11 = mesh->p[v[3]];

    // Compute solid angle PDF for sampling bilinear patch from _ctx_
    // Intersect sample ray with shape geometry
    Ray ray = ctx.SpawnRay(wi);
    pstd::optional<ShapeIntersection> isect = Intersect(ray);
    if (!isect)
        return 0;

    Vector3f v00 = Normalize(p00 - ctx.p()), v10 = Normalize(p10 - ctx.p());
    Vector3f v01 = Normalize(p01 - ctx.p()), v11 = Normalize(p11 - ctx.p());
    if (!IsRectangle(mesh) || mesh->imageDistribution ||
        SphericalQuadArea(v00, v10, v11, v01) <= MinSphericalSampleArea) {
        // Return solid angle PDF for area-sampled bilinear patch
        Float pdf = PDF(isect->intr) * (DistanceSquared(ctx.p(), isect->intr.p()) /
                                        AbsDot(isect->intr.n, -wi));
        return IsInf(pdf) ? 0 : pdf;

    } else {
        // Return PDF for sample in spherical rectangle
        Float pdf = 1 / SphericalQuadArea(v00, v10, v11, v01);
        if (ctx.ns != Normal3f(0, 0, 0)) {
            // Compute $\cos \theta$ weights for rectangle seen from reference point
            pstd::array<Float, 4> w =
                pstd::array<Float, 4>{std::max<Float>(0.01, AbsDot(v00, ctx.ns)),
                                      std::max<Float>(0.01, AbsDot(v10, ctx.ns)),
                                      std::max<Float>(0.01, AbsDot(v01, ctx.ns)),
                                      std::max<Float>(0.01, AbsDot(v11, ctx.ns))};

            Point2f u = InvertSphericalRectangleSample(ctx.p(), p00, p10 - p00, p01 - p00,
                                                       isect->intr.p());
            return BilinearPDF(u, w) * pdf;
        } else
            return pdf;
    }
}

std::string BilinearPatch::ToString() const {
    return StringPrintf("[ BilinearPatch meshIndex: %d blpIndex: %d area: %f ]",
                        meshIndex, blpIndex, area);
}

}
