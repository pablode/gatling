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

#include "MaterialNetworkPatcher.h"

#include <pxr/base/gf/vec4f.h>
#include <pxr/imaging/hd/material.h>
#include <pxr/usd/sdf/assetPath.h>

#include <memory>

const char* ENVVAR_DISABLE_PATCH_USDPREVIEWSURFACE_GLOSSINESS = "HDGATLING_MATPATCH_DISABLE_USDPREVIEWSURFACE_GLOSSINESS";
const char* ENVVAR_DISABLE_PATCH_USDPREVIEWSURFACE_NORMALMAP = "HDGATLING_MATPATCH_DISABLE_USDPREVIEWSURFACE_NORMALMAP";

PXR_NAMESPACE_OPEN_SCOPE

TF_DEFINE_PRIVATE_TOKENS(
  _tokens,
  (ND_UsdPreviewSurface_surfaceshader)
  (ND_UsdUVTexture)
  (glossiness)
  (normal)
  (bias)
  (scale)
);

void _PatchUsdTypes(HdMaterialNetwork2& network)
{
  for (auto& pathNodePair : network.nodes)
  {
    HdMaterialNode2& node = pathNodePair.second;

    auto& parameters = node.parameters;

    for (auto& tokenValuePair : parameters)
    {
      VtValue& value = tokenValuePair.second;

      // Workaround for HdMtlxConvertToString not handling the TfToken type:
      // https://github.com/PixarAnimationStudios/USD/blob/3abc46452b1271df7650e9948fef9f0ce602e3b2/pxr/imaging/hdMtlx/hdMtlx.cpp#L117
      if (value.IsHolding<TfToken>())
      {
        value = value.Cast<std::string>();
      }

      // When serializing the network to a MaterialX document again, the SdfAssetPath
      // gets replaced by its non-resolved path and we don't have any other way of resolving
      // it at a later point in time, since this is done by the Sdf/Ar layer.
      if (value.IsHolding<SdfAssetPath>())
      {
        value = value.UncheckedGet<SdfAssetPath>().GetResolvedPath();
      }
    }
  }
}

// Some of Sketchfab's auto-converted assets encode the roughness on the UsdPreviewSurface
// node with a 'glossiness' input. See "Screen Space Reflection Demo: Follmann 2.OG" scene:
// https://sketchfab.com/3d-models/screen-space-reflection-demo-follmann-2og-6164eed28c464c94be8f5268240dc864
void _PatchUsdPreviewSurfaceGlossiness(HdMaterialNetwork2& network)
{
  for (auto& pathNodePair : network.nodes)
  {
    HdMaterialNode2& node = pathNodePair.second;

    if (node.nodeTypeId != _tokens->ND_UsdPreviewSurface_surfaceshader)
    {
      return;
    }

    auto& parameters = node.parameters;

    auto glossinessParam = parameters.find(_tokens->glossiness);
    if (glossinessParam == parameters.end())
    {
      return;
    }

    VtValue value = glossinessParam->second;
    if (!value.IsHolding<float>())
    {
      return;
    }

    TF_WARN("patching UsdPreviewSurface:glossiness input (set %s to disable)",
      ENVVAR_DISABLE_PATCH_USDPREVIEWSURFACE_GLOSSINESS);

    float glossiness = value.UncheckedGet<float>();
    float roughness = 1.0f - glossiness;

    parameters[TfToken("roughness")] = roughness;
    parameters.erase(glossinessParam);
  }
}

// J CUBE's Maneki asset tries to read a normal map using a UsdUVTexture node,
// however the bias and scale parameters which usually convert the [0, 1] image
// values to [-1, 1] vectors are missing: https://j-cube.jp/solutions/multiverse/assets
void _PatchUsdPreviewSurfaceNormalInputConnection(HdMaterialNetwork2& network, HdMaterialConnection2& connection)
{
  HdMaterialNode2& upstreamNode = network.nodes[connection.upstreamNode];

  if (upstreamNode.nodeTypeId != _tokens->ND_UsdUVTexture)
  {
    return;
  }

  auto& upstreamNodeParams = upstreamNode.parameters;
  bool hasScale = upstreamNodeParams.find(_tokens->scale) != upstreamNodeParams.end();
  bool hasBias = upstreamNodeParams.find(_tokens->bias) != upstreamNodeParams.end();

  if (hasScale || hasBias)
  {
    return;
  }

  TF_WARN("patching UsdPreviewSurface:normal to have scaled and biased reader (set %s to disable)",
    ENVVAR_DISABLE_PATCH_USDPREVIEWSURFACE_NORMALMAP);

  upstreamNodeParams[_tokens->scale] = GfVec4f(2.0f, 2.0f, 2.0f, 1.0f);
  upstreamNodeParams[_tokens->bias] = GfVec4f(-1.0f, -1.0f, -1.0f, 0.0f);
}

void _PatchUsdPreviewSurfaceNormalMap(HdMaterialNetwork2& network)
{
  for (auto& pathNodePair : network.nodes)
  {
    HdMaterialNode2& node = pathNodePair.second;
    if (node.nodeTypeId != _tokens->ND_UsdPreviewSurface_surfaceshader)
    {
      continue;
    }

    auto& inputs = node.inputConnections;

    auto normalInputIt = inputs.find(_tokens->normal);
    if (normalInputIt == inputs.end())
    {
      continue;
    }

    auto& connections = normalInputIt->second;
    for (HdMaterialConnection2& connection : connections)
    {
      _PatchUsdPreviewSurfaceNormalInputConnection(network, connection);
    }
  }
}

void MaterialNetworkPatcher::Patch(HdMaterialNetwork2& network)
{
  _PatchUsdTypes(network);

  if (!getenv(ENVVAR_DISABLE_PATCH_USDPREVIEWSURFACE_GLOSSINESS))
  {
    _PatchUsdPreviewSurfaceGlossiness(network);
  }

  if (!getenv(ENVVAR_DISABLE_PATCH_USDPREVIEWSURFACE_NORMALMAP))
  {
    _PatchUsdPreviewSurfaceNormalMap(network);
  }
}

PXR_NAMESPACE_CLOSE_SCOPE
