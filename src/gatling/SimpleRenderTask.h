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

#include <pxr/pxr.h>
#include <pxr/imaging/hd/task.h>
#include <pxr/imaging/hd/renderPass.h>
#include <pxr/imaging/hd/renderPassState.h>

PXR_NAMESPACE_OPEN_SCOPE

class SimpleRenderTask final : public HdTask
{
public:
  SimpleRenderTask(const HdRenderPassSharedPtr& renderPass,
                   const HdRenderPassStateSharedPtr& renderPassState,
                   const TfTokenVector& renderTags);

  void Sync(HdSceneDelegate* sceneDelegate,
            HdTaskContext* taskContext,
            HdDirtyBits* dirtyBits) override;

  void Prepare(HdTaskContext* taskContext,
               HdRenderIndex* renderIndex) override;

  void Execute(HdTaskContext* taskContext) override;

  const TfTokenVector& GetRenderTags() const override;

private:
  HdRenderPassSharedPtr m_renderPass;
  HdRenderPassStateSharedPtr m_renderPassState;
  TfTokenVector m_renderTags;
};

PXR_NAMESPACE_CLOSE_SCOPE
