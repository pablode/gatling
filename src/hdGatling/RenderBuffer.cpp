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

#include "RenderBuffer.h"

#include <pxr/base/gf/vec3i.h>

#include <gi.h>

PXR_NAMESPACE_OPEN_SCOPE

HdGatlingRenderBuffer::HdGatlingRenderBuffer(const SdfPath& id)
  : HdRenderBuffer(id)
{
  m_isMapped = false;
  m_isConverged = false;
  m_bufferMem = nullptr;
}

HdGatlingRenderBuffer::~HdGatlingRenderBuffer()
{
}

bool HdGatlingRenderBuffer::Allocate(const GfVec3i& dimensions,
                                     HdFormat format,
                                     bool multiSampled)
{
  if (m_renderBuffer)
  {
    giDestroyRenderBuffer(m_renderBuffer);
    m_renderBuffer = nullptr;
  }

  if (dimensions[2] != 1)
  {
    return false;
  }

  m_width = dimensions[0];
  m_height = dimensions[1];
  m_format = format;
  m_isMultiSampled = multiSampled;

  size_t size = m_width * m_height * HdDataSizeOfFormat(m_format);

  m_bufferMem = realloc(m_bufferMem, size);

  if (!m_bufferMem)
  {
    return false;
  }

  m_renderBuffer = giCreateRenderBuffer(m_width, m_height);

  return m_renderBuffer != nullptr;
}

unsigned int HdGatlingRenderBuffer::GetWidth() const
{
  return m_width;
}

unsigned int HdGatlingRenderBuffer::GetHeight() const
{
  return m_height;
}

unsigned int HdGatlingRenderBuffer::GetDepth() const
{
  return 1u;
}

HdFormat HdGatlingRenderBuffer::GetFormat() const
{
  return m_format;
}

bool HdGatlingRenderBuffer::IsMultiSampled() const
{
  return m_isMultiSampled;
}

bool HdGatlingRenderBuffer::IsConverged() const
{
  return m_isConverged;
}

void HdGatlingRenderBuffer::SetConverged(bool converged)
{
  m_isConverged = converged;
}

void* HdGatlingRenderBuffer::Map()
{
  m_isMapped = true;

  return m_bufferMem;
}

bool HdGatlingRenderBuffer::IsMapped() const
{
  return m_isMapped;
}

GiRenderBuffer* HdGatlingRenderBuffer::GetGiRenderBuffer() const
{
  return m_renderBuffer;
}

void HdGatlingRenderBuffer::Unmap()
{
  m_isMapped = false;
}

void HdGatlingRenderBuffer::Resolve()
{
}

void HdGatlingRenderBuffer::_Deallocate()
{
  if (m_renderBuffer)
  {
    giDestroyRenderBuffer(m_renderBuffer);
  }

  free(m_bufferMem);
}

PXR_NAMESPACE_CLOSE_SCOPE
