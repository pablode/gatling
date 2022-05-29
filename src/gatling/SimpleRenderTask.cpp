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

#include "SimpleRenderTask.h"

PXR_NAMESPACE_OPEN_SCOPE

SimpleRenderTask::SimpleRenderTask(const HdRenderPassSharedPtr& renderPass,
                                   const HdRenderPassStateSharedPtr& renderPassState,
                                   const TfTokenVector& renderTags)
  : HdTask(SdfPath::EmptyPath())
  , m_renderPass(renderPass)
  , m_renderPassState(renderPassState)
  , m_renderTags(renderTags)
{
}

void SimpleRenderTask::Sync(HdSceneDelegate* sceneDelegate,
                            HdTaskContext* taskContext,
                            HdDirtyBits* dirtyBits)
{
  TF_UNUSED(sceneDelegate);
  TF_UNUSED(taskContext);

  m_renderPass->Sync();

  *dirtyBits = HdChangeTracker::Clean;
}

void SimpleRenderTask::Prepare(HdTaskContext* taskContext,
                               HdRenderIndex* renderIndex)
{
  TF_UNUSED(taskContext);

  const HdResourceRegistrySharedPtr& resourceRegistry = renderIndex->GetResourceRegistry();
  m_renderPassState->Prepare(resourceRegistry);
}

void SimpleRenderTask::Execute(HdTaskContext* taskContext)
{
  TF_UNUSED(taskContext);

  m_renderPass->Execute(m_renderPassState, m_renderTags);
}

const TfTokenVector& SimpleRenderTask::GetRenderTags() const
{
  return m_renderTags;
}

PXR_NAMESPACE_CLOSE_SCOPE
