#include "Mesh.h"

#include <pxr/imaging/hd/meshUtil.h>
#include <pxr/imaging/hd/vertexAdjacency.h>
#include <pxr/imaging/hd/smoothNormals.h>
#include <pxr/imaging/hd/instancer.h>

PXR_NAMESPACE_OPEN_SCOPE

HdGatlingMesh::HdGatlingMesh(const SdfPath& id)
  : HdMesh(id)
  , m_prototypeTransform(1.0)
  , m_color(0.0, 0.0, 0.0)
{
}

HdGatlingMesh::~HdGatlingMesh()
{
}

void HdGatlingMesh::Sync(HdSceneDelegate* sceneDelegate,
                         HdRenderParam* renderParam,
                         HdDirtyBits* dirtyBits,
                         const TfToken& reprToken)
{
  TF_UNUSED(renderParam);
  TF_UNUSED(reprToken);

  HdRenderIndex& renderIndex = sceneDelegate->GetRenderIndex();

  if ((*dirtyBits & HdChangeTracker::DirtyInstancer) |
      (*dirtyBits & HdChangeTracker::DirtyInstanceIndex))
  {
    HdDirtyBits dirtyBitsCopy = *dirtyBits;

    _UpdateInstancer(sceneDelegate, &dirtyBitsCopy);

    const SdfPath& instancerId = GetInstancerId();

    HdInstancer::_SyncInstancerAndParents(renderIndex, instancerId);
  }

  const SdfPath& id = GetId();

  if (*dirtyBits & HdChangeTracker::DirtyMaterialId)
  {
    HdChangeTracker& changeTracker = renderIndex.GetChangeTracker();
    const SdfPath& materialId = sceneDelegate->GetMaterialId(id);

    _SetMaterialId(changeTracker, materialId);
  }

  if (*dirtyBits & HdChangeTracker::DirtyTransform)
  {
    m_prototypeTransform = sceneDelegate->GetTransform(id);
  }

  bool pullGeometry =
    (*dirtyBits & HdChangeTracker::DirtyPoints) |
    (*dirtyBits & HdChangeTracker::DirtyNormals) |
    (*dirtyBits & HdChangeTracker::DirtyTopology);

  *dirtyBits = HdChangeTracker::Clean;

  if (!pullGeometry)
  {
    return;
  }

  m_faces.clear();
  m_points.clear();
  m_normals.clear();

  _PullGeometry(sceneDelegate);
}

void HdGatlingMesh::_PullGeometry(HdSceneDelegate* sceneDelegate)
{
  HdMeshTopology topology = GetMeshTopology(sceneDelegate);
  const VtIntArray& faceVertexCounts = topology.GetFaceVertexCounts();
  const VtIntArray& faceVertexIndices = topology.GetFaceVertexIndices();

  VtVec3fArray points;
  VtVec3fArray normals;
  _PullPrimvars(sceneDelegate, points, normals, m_color);

  int faceVertexIndexOffset = 0;
  int unsupportedFaceCount = 0;
  std::vector<int> remappedVertexIndices(faceVertexIndices.size(), -1);

  for (int i = 0; i < topology.GetNumFaces(); i++)
  {
    int faceVertexCount = faceVertexCounts[i];

    if (faceVertexCount != 3 && faceVertexCount != 4)
    {
      unsupportedFaceCount++;

      faceVertexIndexOffset += faceVertexCount;

      continue;
    }

    int newFaceVertexIndices[4];

    for (int j = 0; j < faceVertexCount; j++)
    {
      int v_i = faceVertexIndices[faceVertexIndexOffset + j];

      if (remappedVertexIndices[v_i] == -1)
      {
        remappedVertexIndices[v_i] = m_points.size();

        GfVec3f normal = normals[v_i];
        normal.Normalize();
        m_normals.push_back(normal);
        m_points.push_back(points[v_i]);
      }

      newFaceVertexIndices[j] = remappedVertexIndices[v_i];
    }

    faceVertexIndexOffset += faceVertexCount;

    GfVec3i f_0(
      newFaceVertexIndices[0],
      newFaceVertexIndices[1],
      newFaceVertexIndices[2]
    );
    m_faces.push_back(f_0);

    if (faceVertexCount == 3)
    {
      continue;
    }

    GfVec3i f_1(
      newFaceVertexIndices[0],
      newFaceVertexIndices[2],
      newFaceVertexIndices[3]
    );
    m_faces.push_back(f_1);
  }

  if (unsupportedFaceCount > 0)
  {
    const SdfPath& id = GetId();
    const std::string& idStr = id.GetString();
    TF_WARN(TfStringPrintf("Mesh %s has %i unsupported faces", idStr.c_str(), unsupportedFaceCount));
  }
}

bool HdGatlingMesh::_FindPrimvar(HdSceneDelegate* sceneDelegate,
                                 TfToken primvarName,
                                 HdInterpolation& interpolation) const
{
  HdInterpolation interpolations[] = {
    HdInterpolation::HdInterpolationVertex,
    HdInterpolation::HdInterpolationFaceVarying,
    HdInterpolation::HdInterpolationConstant,
    HdInterpolation::HdInterpolationUniform,
    HdInterpolation::HdInterpolationVarying,
    HdInterpolation::HdInterpolationInstance
  };

  for (HdInterpolation i : interpolations)
  {
    const auto& primvarDescs = GetPrimvarDescriptors(sceneDelegate, i);

    for (const HdPrimvarDescriptor& primvar : primvarDescs)
    {
      if (primvar.name == primvarName)
      {
        interpolation = i;

        return true;
      }
    }
  }

  return false;
}

void HdGatlingMesh::_PullPrimvars(HdSceneDelegate* sceneDelegate,
                                  VtVec3fArray& points,
                                  VtVec3fArray& normals,
                                  GfVec3f& color) const
{
  const SdfPath& id = GetId();

  // Handle points.
  HdInterpolation pointInterpolation;
  bool foundPoints = _FindPrimvar(sceneDelegate,
                                  HdTokens->points,
                                  pointInterpolation);

  if (!foundPoints)
  {
    TF_RUNTIME_ERROR("Points primvar not found!");
    return;
  }
  else if (pointInterpolation != HdInterpolation::HdInterpolationVertex)
  {
    TF_RUNTIME_ERROR("Points primvar is not vertex-interpolated!");
    return;
  }

  VtValue boxedPoints = sceneDelegate->Get(id, HdTokens->points);
  points = boxedPoints.Get<VtVec3fArray>();

  // Handle color.
  HdInterpolation colorInterpolation;
  bool foundColor = _FindPrimvar(sceneDelegate,
                                 HdTokens->displayColor,
                                 colorInterpolation);

  if (foundColor && colorInterpolation == HdInterpolation::HdInterpolationConstant)
  {
    VtValue boxedColors = sceneDelegate->Get(id, HdTokens->displayColor);
    const VtVec3fArray& colors = boxedColors.Get<VtVec3fArray>();
    color = colors[0];
  }

  // Handle normals.
  HdInterpolation normalInterpolation;
  bool foundNormals = _FindPrimvar(sceneDelegate,
                                   HdTokens->normals,
                                   normalInterpolation);

  if (foundNormals && normalInterpolation == HdInterpolation::HdInterpolationVertex)
  {
    VtValue boxedNormals = sceneDelegate->Get(id, HdTokens->normals);
    normals = boxedNormals.Get<VtVec3fArray>();
    return;
  }

  // If normals are not per vertex, their indices are used for subdivision:
  // "For faceVarying primvars, however, indexing serves a higher purpose (and should be used only for this purpose,
  // since renderers and OpenSubdiv will assume it) of establishing a surface topology for the primvar."
  // https://graphics.pixar.com/usd/docs/api/class_usd_geom_primvar.html
  HdMeshTopology topology = GetMeshTopology(sceneDelegate);

  Hd_VertexAdjacency adjacency;
  adjacency.BuildAdjacencyTable(&topology);
  normals = Hd_SmoothNormals::ComputeSmoothNormals(&adjacency, points.size(), points.data());
}

const TfTokenVector BUILTIN_PRIMVAR_NAMES =
{
  HdTokens->points,
  HdTokens->normals
};

const TfTokenVector& HdGatlingMesh::GetBuiltinPrimvarNames() const
{
  return BUILTIN_PRIMVAR_NAMES;
}

const std::vector<GfVec3f>& HdGatlingMesh::GetPoints() const
{
  return m_points;
}

const std::vector<GfVec3f>& HdGatlingMesh::GetNormals() const
{
  return m_normals;
}

const std::vector<GfVec3i>& HdGatlingMesh::GetFaces() const
{
  return m_faces;
}

const GfMatrix4d& HdGatlingMesh::GetPrototypeTransform() const
{
  return m_prototypeTransform;
}

const GfVec3f& HdGatlingMesh::GetColor() const
{
  return m_color;
}

HdDirtyBits HdGatlingMesh::GetInitialDirtyBitsMask() const
{
  return HdChangeTracker::DirtyPoints |
         HdChangeTracker::DirtyNormals |
         HdChangeTracker::DirtyTopology |
         HdChangeTracker::DirtyInstancer |
         HdChangeTracker::DirtyInstanceIndex |
         HdChangeTracker::DirtyTransform |
         HdChangeTracker::DirtyMaterialId;
}

HdDirtyBits HdGatlingMesh::_PropagateDirtyBits(HdDirtyBits bits) const
{
  return bits;
}

void HdGatlingMesh::_InitRepr(const TfToken& reprName,
                              HdDirtyBits *dirtyBits)
{
  TF_UNUSED(reprName);
  TF_UNUSED(dirtyBits);
}

PXR_NAMESPACE_CLOSE_SCOPE
