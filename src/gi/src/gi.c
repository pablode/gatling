#include "gi.h"

int giInitialize()
{
}

void giTerminate()
{
}

int giCreateSceneCache(struct gi_scene_cache** cache)
{
}

void giDestroySceneCache(struct gi_scene_cache* cache)
{
}

int giPreprocess(const struct gi_preprocess_params* params,
                 struct gi_scene_cache* scene_cache)
{
}

int giRender(const struct gi_render_params* params,
             float* rgba_img)
{
}
