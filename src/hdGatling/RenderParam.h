//
// Copyright (C) 2023 Pablo Delgado Krämer
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

#include <pxr/imaging/hd/renderDelegate.h>

struct GiDomeLight;

PXR_NAMESPACE_OPEN_SCOPE

class HdGatlingRenderParam final : public HdRenderParam
{
public:
  void AddDomeLight(GiDomeLight* domeLight);

  void SetDomeLightOverride(GiDomeLight* domeLight);

  void RemoveDomeLight(GiDomeLight* domeLight);

  GiDomeLight* ActiveDomeLight() const;

public:
  void SetMeshInstancesDirty(bool dirty);

  bool GetMeshInstancesDirty() const;

private:
  std::vector<GiDomeLight*> m_domeLights;
  GiDomeLight* m_domeLightOverride = nullptr;
  bool m_meshInstancesDirty = false;
};

PXR_NAMESPACE_CLOSE_SCOPE
