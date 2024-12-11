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

#include "renderBuffer.h"

#include <pxr/base/gf/vec3i.h>

#include <gtl/gi/Gi.h>

PXR_NAMESPACE_OPEN_SCOPE

HdGatlingRenderBuffer::HdGatlingRenderBuffer(const SdfPath& id)
  : HdRenderBuffer(id)
{
  _isConverged = false;
}

HdGatlingRenderBuffer::~HdGatlingRenderBuffer()
{
}

bool HdGatlingRenderBuffer::Allocate(const GfVec3i& dimensions,
                                     HdFormat format,
                                     bool multiSampled)
{
  if (_renderBuffer)
  {
    giDestroyRenderBuffer(_renderBuffer);
    _renderBuffer = nullptr;
  }

  if (format != HdFormatFloat32Vec4)
  {
    TF_RUNTIME_ERROR("Unsupported render buffer format!");
    return false;
  }

  if (dimensions[2] != 1)
  {
    TF_RUNTIME_ERROR("3D render buffers not supported!");
    return false;
  }

  _width = dimensions[0];
  _height = dimensions[1];
  _format = format;
  _isMultiSampled = multiSampled;

  _renderBuffer = giCreateRenderBuffer(_width, _height);
  if (!_renderBuffer)
  {
    TF_RUNTIME_ERROR("Failed to create render buffer!");
    return false;
  }

  return true;
}

unsigned int HdGatlingRenderBuffer::GetWidth() const
{
  return _width;
}

unsigned int HdGatlingRenderBuffer::GetHeight() const
{
  return _height;
}

unsigned int HdGatlingRenderBuffer::GetDepth() const
{
  return 1u;
}

HdFormat HdGatlingRenderBuffer::GetFormat() const
{
  return _format;
}

bool HdGatlingRenderBuffer::IsMultiSampled() const
{
  return _isMultiSampled;
}

bool HdGatlingRenderBuffer::IsConverged() const
{
  return _isConverged;
}

void HdGatlingRenderBuffer::SetConverged(bool converged)
{
  _isConverged = converged;
}

void* HdGatlingRenderBuffer::Map()
{
  return _renderBuffer ? giGetRenderBufferMem(_renderBuffer) : nullptr;
}

bool HdGatlingRenderBuffer::IsMapped() const
{
  return bool(_renderBuffer);
}

GiRenderBuffer* HdGatlingRenderBuffer::GetGiRenderBuffer() const
{
  return _renderBuffer;
}

void HdGatlingRenderBuffer::Unmap()
{
}

void HdGatlingRenderBuffer::Resolve()
{
}

void HdGatlingRenderBuffer::_Deallocate()
{
  if (_renderBuffer)
  {
    giDestroyRenderBuffer(_renderBuffer);
  }
}

PXR_NAMESPACE_CLOSE_SCOPE
