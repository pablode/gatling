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
