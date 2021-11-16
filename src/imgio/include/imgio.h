#ifndef IMGIO_H
#define IMGIO_H

#include "error_codes.h"
#include "img.h"

int imgio_load_img(const char* file_path,
                   struct imgio_img* img);

void imgio_free_img(struct imgio_img* img);

#endif
