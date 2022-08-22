//
// Copyright (C) 2019-2022 Pablo Delgado Krämer
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
//

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
                   std::vector<const gi_material*>& materials);

  void _ConstructGiCamera(const HdGatlingCamera& camera, gi_camera& giCamera) const;

  void _ClearColorMaterials();

private:
  const HdRenderSettingsMap& m_settings;
  gi_material* m_defaultMaterial;
  std::vector<gi_material*> m_colorMaterials;
  bool m_isConverged;
  uint32_t m_lastSceneStateVersion;
  uint32_t m_lastRenderSettingsVersion;
  GfVec4f m_lastBackgroundColor;
  gi_aov_id m_lastAovId;
  gi_geom_cache* m_geomCache;
  gi_shader_cache* m_shaderCache;
  GfMatrix4d m_rootMatrix;
};

PXR_NAMESPACE_CLOSE_SCOPE
