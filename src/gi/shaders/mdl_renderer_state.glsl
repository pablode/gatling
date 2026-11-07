#ifndef MDL_RENDERER_STATE
#define MDL_RENDERER_STATE

struct mdl_renderer_state
{
  uvec3 hitIndices;
  vec2 hitBarycentrics;

  uint64_t sceneDataBufferAddress;
  uint64_t paramBufferAddress;
  uint sceneDataInfos[MAX_SCENE_DATA_ID];
};

#endif
