#include "RenderBuffer.h"

#include <pxr/base/gf/vec3i.h>

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

  return true;
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

void HdGatlingRenderBuffer::Unmap()
{
  m_isMapped = false;
}

void HdGatlingRenderBuffer::Resolve()
{
}

void HdGatlingRenderBuffer::_Deallocate()
{
  free(m_bufferMem);
}

PXR_NAMESPACE_CLOSE_SCOPE
