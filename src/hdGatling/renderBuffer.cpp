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
  _isMapped = false;
  _isConverged = false;
  _bufferMem = nullptr;
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

  if (dimensions[2] != 1)
  {
    return false;
  }

  _width = dimensions[0];
  _height = dimensions[1];
  _format = format;
  _isMultiSampled = multiSampled;

  size_t size = _width * _height * HdDataSizeOfFormat(_format);

  _bufferMem = realloc(_bufferMem, size);

  if (!_bufferMem)
  {
    return false;
  }

  _renderBuffer = giCreateRenderBuffer(_width, _height);

  return _renderBuffer != nullptr;
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
  _isMapped = true;

  return _bufferMem;
}

bool HdGatlingRenderBuffer::IsMapped() const
{
  return _isMapped;
}

GiRenderBuffer* HdGatlingRenderBuffer::GetGiRenderBuffer() const
{
  return _renderBuffer;
}

void HdGatlingRenderBuffer::Unmap()
{
  _isMapped = false;
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

  free(_bufferMem);
}

PXR_NAMESPACE_CLOSE_SCOPE
