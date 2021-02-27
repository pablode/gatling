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
