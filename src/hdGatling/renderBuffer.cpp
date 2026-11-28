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

#if defined(_WIN32) && !defined(VK_USE_PLATFORM_WIN32_KHR)
#define VK_USE_PLATFORM_WIN32_KHR
#endif

#include <pxr/base/gf/vec3i.h>
#include <pxr/base/tf/stringUtils.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/hd/types.h>
#include <pxr/imaging/hgi/hgi.h>
#include <pxr/imaging/hgi/texture.h>

#if defined(HDGATLING_HGI_INTEROP_VULKAN)
#include <pxr/imaging/hgiVulkan/device.h>
#include <pxr/imaging/hgiVulkan/hgi.h>
#include <pxr/imaging/hgiVulkan/texture.h>
#endif

#include <gtl/gi/Gi.h>

PXR_NAMESPACE_OPEN_SCOPE

using namespace gtl;

namespace
{
  static std::map<HdFormat, GiRenderBufferFormat> s_supportedRenderBufferFormats = {
    { HdFormatInt32, GiRenderBufferFormat::Int32 },
    { HdFormatFloat32, GiRenderBufferFormat::Float32 },
    { HdFormatFloat32Vec4, GiRenderBufferFormat::Float32Vec4 }
  };

  static std::map<HdFormat, HgiFormat> s_supportedHgiRenderBufferFormats = {
    { HdFormatInt32, HgiFormatInt32 },
    { HdFormatFloat32, HgiFormatFloat32 },
    { HdFormatFloat32Vec4, HgiFormatFloat32Vec4 }
  };

  // Hydra render buffers are named after the AOV they hold, either as
  // <controller>/aov_<name> or <engine>/AOVs/<name>.
  static bool _IsDepthAov(const SdfPath& id)
  {
    std::string name = id.GetElementString();

    if (TfStringStartsWith(name, "aov_"))
    {
      name = name.substr(4);
    }

    return name == HdAovTokens->depth.GetString();
  }

  static HgiTextureHandle _CreateHgiTexture(Hgi* hgi, const HgiTextureDesc& desc)
  {
#if defined(HDGATLING_HGI_INTEROP_VULKAN)
    auto* hgiVulkan = static_cast<HgiVulkan*>(hgi);
    return hgiVulkan->CreateTextureForInterop(desc, /*optimalTiling=*/ true);
#else
    return hgi->CreateTexture(desc);
#endif
  }

#if defined(HDGATLING_HGI_INTEROP_VULKAN)
  static bool _GetVulkanExternalImageDesc(Hgi* hgi, const HgiTextureHandle& texture, GiExternalImageDesc* desc)
  {
    auto* hgiVulkan = static_cast<HgiVulkan*>(hgi);
    auto* hgiVulkanTexture = static_cast<HgiVulkanTexture*>(texture.Get());

    HgiVulkanDevice* device = hgiVulkanTexture->GetDevice();
    VmaAllocationInfo2 allocationInfo = hgiVulkanTexture->GetAllocationInfo();

    desc->memoryOffset = allocationInfo.allocationInfo.offset;
    desc->memorySize = allocationInfo.blockSize;
    desc->memoryTypeIndex = allocationInfo.allocationInfo.memoryType;
    desc->memoryDedicated = allocationInfo.dedicatedMemory;

#if defined(_WIN32)
    desc->memoryHandle = (uint64_t) device->GetWin32HandleForMemory(allocationInfo.allocationInfo.deviceMemory);
    return desc->memoryHandle != 0;
#else
    if (!device->vkGetMemoryFdKHR)
    {
      return false;
    }

    VkMemoryGetFdInfoKHR getInfo = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
      .pNext = nullptr,
      .memory = allocationInfo.allocationInfo.deviceMemory,
      .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };

    int fd = -1;
    if (device->vkGetMemoryFdKHR(device->GetVulkanDevice(), &getInfo, &fd) != VK_SUCCESS)
    {
      return false;
    }

    desc->memoryHandle = (uint64_t) fd;
    return true;
#endif
  }
#endif

  static bool _GetExternalImageDesc(Hgi* hgi, const HgiTextureHandle& texture, uint32_t width, uint32_t height,
                                    GiRenderBufferFormat format, GiExternalImageDesc* desc)
  {
    desc->width = width;
    desc->height = height;
    desc->format = format;

#if defined(HDGATLING_HGI_INTEROP_METAL)
    TF_UNUSED(hgi);
    desc->nativeTexture = texture->GetRawResource();
    return desc->nativeTexture != 0;
#elif defined(HDGATLING_HGI_INTEROP_VULKAN)
    return _GetVulkanExternalImageDesc(hgi, texture, desc);
#else
    return false;
#endif
  }
}

HdGatlingRenderBuffer::HdGatlingRenderBuffer(const SdfPath& id, Hgi* hgi)
  : HdRenderBuffer(id)
  , _isConverged(false)
  , _hgi(hgi)
{
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

  if (_hgiTexture)
  {
    _hgi->DestroyTexture(&_hgiTexture);
  }

  auto it = s_supportedRenderBufferFormats.find(format);
  if (it == s_supportedRenderBufferFormats.end())
  {
    TF_RUNTIME_ERROR("Unsupported render buffer format!");
    return false;
  }

  if (dimensions[2] != 1)
  {
    TF_RUNTIME_ERROR("3D render buffers not supported!");
    return false;
  }

  if (dimensions[0] == 0 || dimensions[1] == 0)
  {
    TF_RUNTIME_ERROR("Can't allocate empty render buffer!");
    return false;
  }

  _width = dimensions[0];
  _height = dimensions[1];
  _format = format;
  _isMultiSampled = multiSampled;

  // Hydra uses the depth AOV as a depth attachment, which a float storage image
  // is not. It therefore stays in host memory and is converted to a depth
  // texture by HdxAovInputTask.
  bool useHgiTexture = _hgi && !_IsDepthAov(GetId());

  if (useHgiTexture)
  {
    HgiTextureDesc desc;
    desc.debugName = "hdGatling render buffer";
    desc.usage = HgiTextureUsageBitsShaderRead | HgiTextureUsageBitsShaderWrite;
    desc.format = s_supportedHgiRenderBufferFormats.at(_format);
    desc.type = HgiTextureType2D;
    desc.dimensions = GfVec3i(_width, _height, 1);
    desc.layerCount = 1;
    desc.mipLevels = 1;
    desc.sampleCount = HgiSampleCount1;

    _hgiTexture = _CreateHgiTexture(_hgi, desc);

    GiExternalImageDesc externalImageDesc = {};
    if (_hgiTexture && _GetExternalImageDesc(_hgi, _hgiTexture, _width, _height, it->second, &externalImageDesc))
    {
      _renderBuffer = giCreateRenderFromExternal(externalImageDesc);
    }

    if (!_renderBuffer && _hgiTexture)
    {
      TF_WARN("Failed to create GPU interop render buffer; falling back to host memory");
      _hgi->DestroyTexture(&_hgiTexture);
    }
  }

  if (!_renderBuffer)
  {
    _renderBuffer = giCreateRenderBuffer(_width, _height, it->second);
  }

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
  return _renderBuffer ? giMapRenderBuffer(_renderBuffer) : nullptr;
}

bool HdGatlingRenderBuffer::IsMapped() const
{
  return bool(_renderBuffer);
}

GiRenderBuffer* HdGatlingRenderBuffer::GetGiRenderBuffer() const
{
  return _renderBuffer;
}

VtValue HdGatlingRenderBuffer::GetResource(bool multiSampled) const
{
  TF_UNUSED(multiSampled);

  return _hgiTexture ? VtValue(_hgiTexture) : VtValue();
}

void HdGatlingRenderBuffer::Unmap()
{
  if (_renderBuffer)
  {
    giUnmapRenderBuffer(_renderBuffer);
  }
}

void HdGatlingRenderBuffer::Resolve()
{
}

void HdGatlingRenderBuffer::_Deallocate()
{
  if (_renderBuffer)
  {
    giDestroyRenderBuffer(_renderBuffer);
    _renderBuffer = nullptr;
  }

  if (_hgiTexture)
  {
    _hgi->DestroyTexture(&_hgiTexture);
  }
}

PXR_NAMESPACE_CLOSE_SCOPE
