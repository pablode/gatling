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

#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/types.h>

#include <gtl/gi/Gi.h>

PXR_NAMESPACE_OPEN_SCOPE

bool HdGatlingIsPrimvarTypeSupported(const VtValue& value);

gtl::GiPrimvarType HdGatlingGetGiPrimvarType(HdType type);

void HdGatlingConvertVtBoolArrayToVtIntArray(VtValue& values);

bool HdGatlingUnboxPRSPrimvars(VtValue& boxedTranslations,
                               VtValue& boxedRotations,
                               VtValue& boxedScales,
                               VtVec3dArray& translations,
                               VtQuatdArray& rotations,
                               VtVec3dArray& scales);

void HdGatlingPRSToTransforms(const VtIntArray& indices,
                              const GfMatrix4d& rootTransform,
                              const VtMatrix4dArray& instanceTransforms,
                              const VtVec3dArray& translations,
                              const VtQuatdArray& rotations,
                              const VtVec3dArray& scales,
                              VtMatrix4fArray& transforms);

PXR_NAMESPACE_CLOSE_SCOPE
