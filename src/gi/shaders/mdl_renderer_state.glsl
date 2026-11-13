#ifndef MDL_RENDERER_STATE
#define MDL_RENDERER_STATE

struct mdl_renderer_state
{
  uvec3 hitIndices;
  vec2 hitBarycentrics;
  uint materialInstance;
  uint64_t textureParameterBufferAddress;

  uint64_t sceneDataInfos[MAX_SCENE_DATA_ID];
};

#endif
