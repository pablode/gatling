#include "RenderPass.h"
#include "RenderBuffer.h"
#include "Camera.h"
#include "Mesh.h"
#include "Instancer.h"
#include "Material.h"
#include "Tokens.h"

#include <pxr/imaging/hd/renderPassState.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/rprim.h>
#include <pxr/base/gf/matrix3d.h>

#include <gi.h>

PXR_NAMESPACE_OPEN_SCOPE

HdGatlingRenderPass::HdGatlingRenderPass(HdRenderIndex* index,
                                         const HdRprimCollection& collection,
                                         const HdRenderSettingsMap& settings)
  : HdRenderPass(index, collection)
  , m_settings(settings)
  , m_isConverged(false)
  , m_lastSceneStateVersion(UINT32_MAX)
  , m_lastRenderSettingsVersion(UINT32_MAX)
  , m_sceneCache(nullptr)
{
  m_defaultMaterial.albedo[0] = 1.0f;
  m_defaultMaterial.albedo[1] = 1.0f;
  m_defaultMaterial.albedo[2] = 1.0f;
  m_defaultMaterial.emission[0] = 0.0f;
  m_defaultMaterial.emission[1] = 0.0f;
  m_defaultMaterial.emission[2] = 0.0f;
}

HdGatlingRenderPass::~HdGatlingRenderPass()
{
  if (m_sceneCache)
  {
    giDestroySceneCache(m_sceneCache);
  }
}

bool HdGatlingRenderPass::IsConverged() const
{
  return m_isConverged;
}

void HdGatlingRenderPass::_BakeMeshInstance(const HdGatlingMesh* mesh,
                                            GfMatrix4d transform,
                                            uint32_t materialIndex,
                                            std::vector<gi_face>& faces,
                                            std::vector<gi_vertex>& vertices) const
{
  GfMatrix4d normalMatrix = transform.GetInverse().GetTranspose();

  const std::vector<GfVec3f>& meshPoints = mesh->GetPoints();
  const std::vector<GfVec3f>& meshNormals = mesh->GetNormals();
  const std::vector<GfVec3i>& meshFaces = mesh->GetFaces();
  TF_VERIFY(meshPoints.size() == meshNormals.size());

  uint32_t vertexOffset = vertices.size();

  for (size_t j = 0; j < meshFaces.size(); j++)
  {
    const GfVec3i& vertexIndices = meshFaces[j];

    gi_face face;
    face.v_i[0] = vertexOffset + vertexIndices[0];
    face.v_i[1] = vertexOffset + vertexIndices[1];
    face.v_i[2] = vertexOffset + vertexIndices[2];
    face.mat_index = materialIndex;

    faces.push_back(face);
  }

  for (size_t j = 0; j < meshPoints.size(); j++)
  {
    const GfVec3f& point = meshPoints[j];
    const GfVec3f& normal = meshNormals[j];

    GfVec3f newPoint = transform.Transform(point);
    GfVec3f newNormal = normalMatrix.TransformDir(normal);
    newNormal.Normalize();

    gi_vertex vertex;
    vertex.pos[0] = newPoint[0];
    vertex.pos[1] = newPoint[1];
    vertex.pos[2] = newPoint[2];
    vertex.norm[0] = newNormal[0];
    vertex.norm[1] = newNormal[1];
    vertex.norm[2] = newNormal[2];

    vertices.push_back(vertex);
  }
}

void HdGatlingRenderPass::_BakeMeshes(HdRenderIndex* renderIndex,
                                      GfMatrix4d rootTransform,
                                      std::vector<gi_vertex>& vertices,
                                      std::vector<gi_face>& faces,
                                      std::vector<gi_material>& materials) const
{
  vertices.clear();
  faces.clear();

  TfHashMap<SdfPath, uint32_t, SdfPath::Hash> materialMapping;
  materialMapping[SdfPath::EmptyPath()] = 0;

  materials.push_back(m_defaultMaterial);

  for (const auto& rprimId : renderIndex->GetRprimIds())
  {
    const HdRprim* rprim = renderIndex->GetRprim(rprimId);

    if (!dynamic_cast<const HdMesh*>(rprim))
    {
      continue;
    }

    const HdGatlingMesh* mesh = dynamic_cast<const HdGatlingMesh*>(rprim);

    VtMatrix4dArray transforms;
    const SdfPath& instancerId = mesh->GetInstancerId();

    if (instancerId.IsEmpty())
    {
      transforms.resize(1);
      transforms[0] = GfMatrix4d(1.0);
    }
    else
    {
      HdInstancer* boxedInstancer = renderIndex->GetInstancer(instancerId);
      HdGatlingInstancer* instancer = dynamic_cast<HdGatlingInstancer*>(boxedInstancer);

      const SdfPath& meshId = mesh->GetId();
      transforms = instancer->ComputeInstanceTransforms(meshId);
    }

    const SdfPath& materialId = mesh->GetMaterialId();
    uint32_t materialIndex = 0;

    if (materialId.IsEmpty() && mesh->HasColor())
    {
      const GfVec3f& color = mesh->GetColor();

      gi_material newMaterial;
      newMaterial.albedo[0] = color[0];
      newMaterial.albedo[1] = color[1];
      newMaterial.albedo[2] = color[2];
      newMaterial.emission[0] = 0.0f;
      newMaterial.emission[1] = 0.0f;
      newMaterial.emission[2] = 0.0f;

      materialIndex = materials.size();
      materials.push_back(newMaterial);
    }
    else if (materialMapping.find(materialId) != materialMapping.end())
    {
      materialIndex = materialMapping[materialId];
    }
    else
    {
      HdSprim* sprim = renderIndex->GetSprim(HdPrimTypeTokens->material, materialId);
      HdGatlingMaterial* material = dynamic_cast<HdGatlingMaterial*>(sprim);

      if (material)
      {
        materialIndex = materials.size();
        materials.push_back(material->GetGiMaterial());
        materialMapping[materialId] = materialIndex;
      }
    }

    const GfMatrix4d& prototypeTransform = mesh->GetPrototypeTransform();

    for (size_t i = 0; i < transforms.size(); i++)
    {
      GfMatrix4d transform = prototypeTransform * transforms[i] * rootTransform;

      _BakeMeshInstance(mesh, transform, materialIndex, faces, vertices);
    }
  }

  printf("#Vertices: %zu\n", vertices.size());
  printf("#Faces: %zu\n", faces.size());
  fflush(stdout);
}

void HdGatlingRenderPass::_ConstructGiCamera(const HdGatlingCamera& camera, gi_camera& giCamera) const
{
  // We transform the scene into camera space at the beginning, so for
  // subsequent camera transforms, we need to 'substract' the initial transform.
  GfMatrix4d absInvViewMatrix = camera.GetTransform();
  GfMatrix4d relViewMatrix = absInvViewMatrix * m_rootMatrix;

  GfVec3d position = relViewMatrix.Transform(GfVec3d(0.0, 0.0, 0.0));
  GfVec3d forward = relViewMatrix.TransformDir(GfVec3d(0.0, 0.0, -1.0));
  GfVec3d up = relViewMatrix.TransformDir(GfVec3d(0.0, 1.0, 0.0));

  forward.Normalize();
  up.Normalize();

  giCamera.position[0] = (float) position[0];
  giCamera.position[1] = (float) position[1];
  giCamera.position[2] = (float) position[2];
  giCamera.forward[0] = (float) forward[0];
  giCamera.forward[1] = (float) forward[1];
  giCamera.forward[2] = (float) forward[2];
  giCamera.up[0] = (float) up[0];
  giCamera.up[1] = (float) up[1];
  giCamera.up[2] = (float) up[2];
  giCamera.vfov = camera.GetVFov();
}

void HdGatlingRenderPass::_Execute(const HdRenderPassStateSharedPtr& renderPassState,
                                   const TfTokenVector& renderTags)
{
  TF_UNUSED(renderTags);

  m_isConverged = false;

  const auto* camera = dynamic_cast<const HdGatlingCamera*>(renderPassState->GetCamera());

  if (!camera)
  {
    return;
  }

  const HdRenderPassAovBindingVector& aovBindings = renderPassState->GetAovBindings();

  if (aovBindings.empty())
  {
    return;
  }

  const HdRenderPassAovBinding* colorAovBinding = nullptr;

  for (const HdRenderPassAovBinding& aovBinding : aovBindings)
  {
    if (aovBinding.aovName != HdAovTokens->color)
    {
      HdGatlingRenderBuffer* renderBuffer = dynamic_cast<HdGatlingRenderBuffer*>(aovBinding.renderBuffer);
      renderBuffer->SetConverged(true);
      continue;
    }

    colorAovBinding = &aovBinding;
  }

  if (!colorAovBinding)
  {
    return;
  }

  HdRenderIndex* renderIndex = GetRenderIndex();
  HdChangeTracker& changeTracker = renderIndex->GetChangeTracker();
  HdRenderDelegate* renderDelegate = renderIndex->GetRenderDelegate();
  HdGatlingRenderBuffer* renderBuffer = dynamic_cast<HdGatlingRenderBuffer*>(colorAovBinding->renderBuffer);

  uint32_t sceneStateVersion = changeTracker.GetSceneStateVersion();
  uint32_t renderSettingsStateVersion = renderDelegate->GetRenderSettingsVersion();

  bool sceneChanged = (sceneStateVersion != m_lastSceneStateVersion);
  bool renderSettingsChanged = (renderSettingsStateVersion != m_lastRenderSettingsVersion);

  if (!sceneChanged && !renderSettingsChanged)
  {
    renderBuffer->SetConverged(true);
    return;
  }

  renderBuffer->SetConverged(false);

  m_lastSceneStateVersion = sceneStateVersion;
  m_lastRenderSettingsVersion = renderSettingsStateVersion;

  if (!m_sceneCache)
  {
    const SdfPath& cameraId = camera->GetId();
    printf("Building scene cache for camera %s\n", cameraId.GetText());
    fflush(stdout);

    std::vector<gi_vertex> vertices;
    std::vector<gi_face> faces;
    std::vector<gi_material> materials;

    // Transform scene into camera space to increase floating point precision.
    GfMatrix4d viewMatrix = camera->GetTransform().GetInverse();

    _BakeMeshes(renderIndex, viewMatrix, vertices, faces, materials);

    gi_preprocess_params preprocessParams;
    preprocessParams.face_count = faces.size();
    preprocessParams.faces = faces.data();
    preprocessParams.vertex_count = vertices.size();
    preprocessParams.vertices = vertices.data();
    preprocessParams.material_count = materials.size();
    preprocessParams.materials = materials.data();

    int32_t result;

    result = giCreateSceneCache(&m_sceneCache);

    TF_VERIFY(result == GI_OK, "Unable to create scene cache.");

    result = giPreprocess(&preprocessParams,
                          m_sceneCache);

    TF_VERIFY(result == GI_OK, "Unable to preprocess scene.");

    m_rootMatrix = viewMatrix;
  }

  gi_camera giCamera;
  _ConstructGiCamera(*camera, giCamera);

  gi_render_params renderParams;
  renderParams.scene_cache = m_sceneCache;
  renderParams.camera = &giCamera;
  renderParams.image_width = renderBuffer->GetWidth();
  renderParams.image_height = renderBuffer->GetHeight();
  renderParams.spp = m_settings.find(HdGatlingSettingsTokens->spp)->second.Get<int>();
  renderParams.max_bounces = m_settings.find(HdGatlingSettingsTokens->max_bounces)->second.Get<int>();
  renderParams.rr_bounce_offset = m_settings.find(HdGatlingSettingsTokens->rr_bounce_offset)->second.Get<int>();
  // Workaround for bug https://github.com/PixarAnimationStudios/USD/issues/913
  VtValue rr_inv_min_term_prob = m_settings.find(HdGatlingSettingsTokens->rr_inv_min_term_prob)->second;
  renderParams.rr_inv_min_term_prob = rr_inv_min_term_prob.CastToTypeid(typeid(double)).Get<double>();

  float* img_data = (float*) renderBuffer->Map();

  int32_t result = giRender(&renderParams,
                            img_data);
  TF_VERIFY(result == GI_OK, "Unable to render scene.");

  renderBuffer->Unmap();
  renderBuffer->SetConverged(true);

  m_isConverged = true;
}

PXR_NAMESPACE_CLOSE_SCOPE
