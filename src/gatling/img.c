#include "img.h"

#include <stdlib.h>
#include <string.h>
#include <png.h>

#include "mmap.h"

typedef struct gatling_img_write_state
{
  uint8_t* enc_data;
  uint64_t size;
}
gatling_img_write_state;

static void gatling_img_write_cb(
  png_structp png_ptr,
  png_bytep data,
  png_size_t size)
{
  gatling_img_write_state* state = (gatling_img_write_state*) png_get_io_ptr(png_ptr);

  uint64_t new_size = state->size + size;

  if (state->enc_data)
  {
    state->enc_data = realloc(state->enc_data, new_size);
  }
  else
  {
    state->enc_data = malloc(new_size);
  }

  if (!state->enc_data)
  {
    png_error(png_ptr, "Out of memory.");
  }

  memcpy(
    (void*) &state->enc_data[state->size],
    (const void*) data,
    (size_t) size
  );

  state->size += size;
}

bool gatling_img_write(
  const uint8_t* data,
  uint32_t width,
  uint32_t height,
  const char* path)
{
  png_structp png_ptr = png_create_write_struct(
    PNG_LIBPNG_VER_STRING,
    NULL,
    NULL,
    NULL
  );

  if (!png_ptr)
  {
    return false;
  }

  png_infop info_ptr = png_create_info_struct(png_ptr);

  if (!info_ptr)
  {
    png_destroy_write_struct(&png_ptr, NULL);
    return false;
  }

  if (setjmp(png_jmpbuf(png_ptr)))
  {
    png_destroy_write_struct(&png_ptr, &info_ptr);
    return false;
  }

  gatling_img_write_state state;
  state.enc_data = NULL;
  state.size = 0;

  png_set_IHDR(
    png_ptr,
    info_ptr,
    width,
    height,
    8,
    PNG_COLOR_TYPE_RGB,
    PNG_INTERLACE_NONE,
    PNG_COMPRESSION_TYPE_DEFAULT,
    PNG_FILTER_TYPE_DEFAULT
  );

  png_set_write_fn(png_ptr, &state, gatling_img_write_cb, NULL);

  png_bytep* row_ptrs = (png_bytep*) malloc(sizeof(png_bytep) * height);

  if (!row_ptrs)
  {
    png_destroy_write_struct(&png_ptr, &info_ptr);
    return false;
  }

  for (uint32_t y = 0; y < height; ++y)
  {
    const uint8_t* row_ptr = &data[y * width * 3];

    row_ptrs[height - y - 1] = (png_bytep) row_ptr;
  }

  png_set_rows(png_ptr, info_ptr, row_ptrs);

  if (setjmp(png_jmpbuf(png_ptr)))
  {
    free(row_ptrs);
    free(state.enc_data);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    return false;
  }

  png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);

  free(row_ptrs);

  png_destroy_write_struct(&png_ptr, &info_ptr);

  gatling_file* file;

  if (!gatling_file_create(path, state.size, &file))
  {
    free(state.enc_data);
    return false;
  }

  void* mapped_mem = gatling_mmap(file, 0, state.size);

  if (!mapped_mem)
  {
    free(state.enc_data);
    return false;
  }

  memcpy(mapped_mem, state.enc_data, state.size);

  gatling_munmap(file, mapped_mem);
  gatling_file_close(file);

  free(state.enc_data);

  return true;
}
