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

#include "RenderParam.h"

PXR_NAMESPACE_OPEN_SCOPE

void HdGatlingRenderParam::AddDomeLight(GiDomeLight* domeLight)
{
  m_domeLights.push_back(domeLight);
}

void HdGatlingRenderParam::SetDomeLightOverride(GiDomeLight* domeLight)
{
  m_domeLightOverride = domeLight;
}

void HdGatlingRenderParam::RemoveDomeLight(GiDomeLight* domeLight)
{
  m_domeLights.erase(std::remove(m_domeLights.begin(), m_domeLights.end(), domeLight), m_domeLights.end());

  if (m_domeLightOverride == domeLight)
  {
    m_domeLightOverride = nullptr;
  }
}

GiDomeLight* HdGatlingRenderParam::ActiveDomeLight() const
{
  if (m_domeLightOverride)
  {
    return m_domeLightOverride;
  }

  return m_domeLights.size() > 0 ? m_domeLights.back() : nullptr;
}

PXR_NAMESPACE_CLOSE_SCOPE
