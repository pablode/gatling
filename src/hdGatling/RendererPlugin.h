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

#include <pxr/imaging/hd/rendererPlugin.h>

#include <memory>

#include "MaterialNetworkTranslator.h"

PXR_NAMESPACE_OPEN_SCOPE

class HdGatlingRendererPlugin final : public HdRendererPlugin
{
public:
  HdGatlingRendererPlugin();

  ~HdGatlingRendererPlugin() override;

public:
  HdRenderDelegate* CreateRenderDelegate() override;

  HdRenderDelegate* CreateRenderDelegate(const HdRenderSettingsMap& settingsMap) override;

  void DeleteRenderDelegate(HdRenderDelegate* renderDelegate) override;

  bool IsSupported() const override;

private:
  std::unique_ptr<MaterialNetworkTranslator> m_translator;
  std::unique_ptr<class UsdzAssetReader> m_usdzAssetReader;
  bool m_isSupported;
};

PXR_NAMESPACE_CLOSE_SCOPE
