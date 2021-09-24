#pragma once

#include <pxr/pxr.h>
#include <pxr/imaging/hd/renderPass.h>
#include <pxr/imaging/hd/renderDelegate.h>

#include <gi.h>

PXR_NAMESPACE_OPEN_SCOPE

class HdGatlingCamera;
class HdGatlingMesh;

class HdGatlingRenderPass final : public HdRenderPass
{
public:
  HdGatlingRenderPass(HdRenderIndex* index,
                      const HdRprimCollection& collection,
                      const HdRenderSettingsMap& settings);

  ~HdGatlingRenderPass() override;

public:
  bool IsConverged() const override;

protected:
  void _Execute(const HdRenderPassStateSharedPtr& renderPassState,
                const TfTokenVector& renderTags) override;

private:
  void _BakeMeshInstance(const HdGatlingMesh* mesh,
                         GfMatrix4d transform,
                         uint32_t materialIndex,
                         std::vector<gi_face>& faces,
                         std::vector<gi_vertex>& vertices) const;

  void _BakeMeshes(HdRenderIndex* renderIndex,
                   GfMatrix4d rootTransform,
                   std::vector<gi_vertex>& vertices,
                   std::vector<gi_face>& faces,
                   std::vector<gi_material>& materials) const;

  void _ConstructGiCamera(const HdGatlingCamera& camera, gi_camera& giCamera) const;

private:
  const HdRenderSettingsMap& m_settings;
  gi_material m_defaultMaterial;
  bool m_isConverged;
  uint32_t m_lastSceneStateVersion;
  uint32_t m_lastRenderSettingsVersion;
  gi_scene_cache* m_sceneCache;
  GfMatrix4d m_rootMatrix;
};

PXR_NAMESPACE_CLOSE_SCOPE
