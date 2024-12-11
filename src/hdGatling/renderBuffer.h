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

#include <pxr/imaging/hd/renderBuffer.h>

namespace gtl
{
  struct GiRenderBuffer;
}

using namespace gtl;

PXR_NAMESPACE_OPEN_SCOPE

class HdGatlingRenderBuffer final : public HdRenderBuffer
{
public:
  HdGatlingRenderBuffer(const SdfPath& id);

  ~HdGatlingRenderBuffer() override;

public:
  bool Allocate(const GfVec3i& dimensions,
                HdFormat format,
                bool multiSamples) override;

public:
  unsigned int GetWidth() const override;

  unsigned int GetHeight() const override;

  unsigned int GetDepth() const override;

  HdFormat GetFormat() const override;

  bool IsMultiSampled() const override;

  GiRenderBuffer* GetGiRenderBuffer() const;

public:
  bool IsConverged() const override;

  void SetConverged(bool converged);

public:
  void* Map() override;

  bool IsMapped() const override;

  void Unmap() override;

  void Resolve() override;

protected:
  void _Deallocate() override;

private:
  uint32_t _width;
  uint32_t _height;
  HdFormat _format;
  bool _isMultiSampled;
  bool _isConverged;
  GiRenderBuffer* _renderBuffer = nullptr;
};

PXR_NAMESPACE_CLOSE_SCOPE
