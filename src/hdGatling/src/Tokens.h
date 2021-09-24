#pragma once

#include <pxr/base/tf/staticTokens.h>

PXR_NAMESPACE_OPEN_SCOPE

#define HD_GATLING_SETTINGS_TOKENS                  \
  ((spp, "spp"))                                    \
  ((max_bounces, "max-bounces"))                    \
  ((rr_bounce_offset, "rr-bounce-offset"))          \
  ((rr_inv_min_term_prob, "rr-inv-min-term-prob"))

TF_DECLARE_PUBLIC_TOKENS(HdGatlingSettingsTokens, HD_GATLING_SETTINGS_TOKENS);

PXR_NAMESPACE_CLOSE_SCOPE
