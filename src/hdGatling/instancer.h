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

#include <pxr/imaging/hd/instancer.h>

#include <gtl/gi/Gi.h>

PXR_NAMESPACE_OPEN_SCOPE

class HdGatlingInstancer final : public HdInstancer
{
public:
  HdGatlingInstancer(HdSceneDelegate* delegate,
                     const SdfPath& id);

  ~HdGatlingInstancer() override;

public:
  VtMatrix4fArray ComputeFlattenedTransforms(const SdfPath& prototypeId);

  std::vector<gtl::GiPrimvarData> ComputeFlattenedPrimvars(const SdfPath& prototypeId);

  void Sync(HdSceneDelegate* sceneDelegate,
            HdRenderParam* renderParam,
            HdDirtyBits* dirtyBits) override;

private:
  std::vector<gtl::GiPrimvarData> MakeGiPrimvars(const SdfPath& prototypeId);

  TfHashMap<TfToken, VtValue, TfToken::HashFunctor> _primvarMap;
};

PXR_NAMESPACE_CLOSE_SCOPE
