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

#include <pxr/imaging/hd/renderPass.h>
#include <pxr/imaging/hd/renderDelegate.h>

#include <gtl/gi/Gi.h>

using namespace gtl;

PXR_NAMESPACE_OPEN_SCOPE

class HdCamera;
class HdGatlingCamera;
class HdGatlingMesh;
class MaterialNetworkCompiler;

class HdGatlingRenderPass final : public HdRenderPass
{
public:
  HdGatlingRenderPass(HdRenderIndex* index,
                      const HdRprimCollection& collection,
                      const HdRenderSettingsMap& settings,
                      GiScene* scene);

  ~HdGatlingRenderPass() override;

public:
  bool IsConverged() const override;

protected:
  void _Execute(const HdRenderPassStateSharedPtr& renderPassState,
                const TfTokenVector& renderTags) override;

private:
  void _ConstructGiCamera(const HdCamera& camera, GiCameraDesc& giCamera, bool clippingEnabled) const;

private:
  GiScene* _scene;
  const HdRenderSettingsMap& _settings;
  bool _isConverged;
  uint32_t _lastSceneStateVersion;
  uint32_t _lastSprimIndexVersion;
  uint32_t _lastRenderSettingsVersion;
  uint32_t _lastVisChangeCount;
  GiAovId _lastAovId;
  GiBvh* _bvh;
  GiShaderCache* _shaderCache;
  GfMatrix4d _rootMatrix;
};

PXR_NAMESPACE_CLOSE_SCOPE
