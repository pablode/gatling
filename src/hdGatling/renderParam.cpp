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

#include "renderParam.h"

PXR_NAMESPACE_OPEN_SCOPE

void HdGatlingRenderParam::AddDomeLight(GiDomeLight* domeLight)
{
  _domeLights.push_back(domeLight);
}

void HdGatlingRenderParam::SetDomeLightOverride(GiDomeLight* domeLight)
{
  _domeLightOverride = domeLight;
}

void HdGatlingRenderParam::RemoveDomeLight(GiDomeLight* domeLight)
{
  _domeLights.erase(std::remove(_domeLights.begin(), _domeLights.end(), domeLight), _domeLights.end());

  if (_domeLightOverride == domeLight)
  {
    _domeLightOverride = nullptr;
  }
}

GiDomeLight* HdGatlingRenderParam::ActiveDomeLight() const
{
  if (_domeLightOverride)
  {
    return _domeLightOverride;
  }

  return _domeLights.size() > 0 ? _domeLights.back() : nullptr;
}

PXR_NAMESPACE_CLOSE_SCOPE
