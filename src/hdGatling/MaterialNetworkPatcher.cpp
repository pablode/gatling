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

#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/imaging/hd/material.h>
#include <pxr/usd/sdf/assetPath.h>

#include <memory>

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
  (isSRGB)
  (sourceColorSpace)
  ((_auto, "auto"))
  (yes)
  (no)
  (sRGB)
  (raw)
  (roughness)
  (specular)
  (specularColor)
  (rgb)
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

void _PatchUsdPreviewSurfaceGlossinessInput(HdMaterialNetwork2& network, std::map<TfToken, std::vector<HdMaterialConnection2>>& inputs)
{
  auto glossinessInput = inputs.find(_tokens->glossiness);
  if (glossinessInput == inputs.end())
  {
    return;
  }

  auto connections = glossinessInput->second;
  for (HdMaterialConnection2& connection : connections)
  {
    HdMaterialNode2& upstreamNode = network.nodes[connection.upstreamNode];

    if (upstreamNode.nodeTypeId != _tokens->ND_UsdUVTexture)
    {
      continue;
    }

    auto& upstreamNodeParams = upstreamNode.parameters;
    bool hasScale = upstreamNodeParams.find(_tokens->scale) != upstreamNodeParams.end();
    bool hasBias = upstreamNodeParams.find(_tokens->bias) != upstreamNodeParams.end();

    if (hasScale || hasBias)
    {
      continue;
    }

    // Map glossiness to roughness as follows:
    // output = textureValue * scale + bias
    upstreamNodeParams[_tokens->scale] = GfVec4f(-1.0f, -1.0f, -1.0f, -1.0f);
    upstreamNodeParams[_tokens->bias] = GfVec4f(1.0f, 1.0f, 1.0f, 0.0f);
  }

  inputs[_tokens->roughness] = connections;
  inputs.erase(glossinessInput);
}

bool _PatchUsdPreviewSurfaceGlossinessParam(std::map<TfToken, VtValue>& params)
{
  auto glossinessParam = params.find(_tokens->glossiness);
  if (glossinessParam == params.end())
  {
    return false;
  }

  VtValue value = glossinessParam->second;
  if (!value.IsHolding<float>())
  {
    return true;
  }

  float glossiness = value.UncheckedGet<float>();
  float roughness = 1.0f - glossiness;

  params[_tokens->roughness] = roughness;
  params.erase(glossinessParam);
  return true;
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
      continue;
    }

    auto& parameters = node.parameters;
    if (_PatchUsdPreviewSurfaceGlossinessParam(parameters))
    {
      continue;
    }

    auto& inputs = node.inputConnections;
    _PatchUsdPreviewSurfaceGlossinessInput(network, inputs);
  }
}

// Blender's USD exporter (3.1+) emits a 'specular' input/param which should be 'specularColor'.
// https://github.com/blender/blender/blob/e1b3d9112730bc3b569ffff732a1558752ded146/source/blender/io/usd/intern/usd_writer_material.cc#L234
void _PatchUsdPreviewSurfaceSpecular(HdMaterialNetwork2& network)
{
  for (auto& pathNodePair : network.nodes)
  {
    HdMaterialNode2& node = pathNodePair.second;

    if (node.nodeTypeId != _tokens->ND_UsdPreviewSurface_surfaceshader)
    {
      continue;
    }

    auto& params = node.parameters;

    // Rename params and change type to Color3
    auto specularParam = params.find(_tokens->specular);
    if (specularParam != params.end())
    {
      if (specularParam->second.IsHolding<float>())
      {
        float specular = specularParam->second.UncheckedGet<float>();
        params[_tokens->specularColor] = GfVec3f(specular);
        params.erase(specularParam);
      }
      continue;
    }

    auto& inputs = node.inputConnections;

    // Rename inputs and change connected UsdUVTexture output from single-channel to 'rgb'
    auto specularInput = inputs.find(_tokens->specular);
    if (specularInput != inputs.end())
    {
      auto connections = specularInput->second;

      for (HdMaterialConnection2& connection : connections)
      {
        HdMaterialNode2& upstreamNode = network.nodes[connection.upstreamNode];

        if (upstreamNode.nodeTypeId != _tokens->ND_UsdUVTexture)
        {
          continue;
        }

        connection.upstreamOutputName = _tokens->rgb;
      }

      inputs[_tokens->specularColor] = connections;
      inputs.erase(specularInput);
    }
  }
}

void _PatchUsdPreviewSurfaceNormalInputConnection(HdMaterialNetwork2& network, HdMaterialConnection2& connection)
{
  HdMaterialNode2& upstreamNode = network.nodes[connection.upstreamNode];

  if (upstreamNode.nodeTypeId != _tokens->ND_UsdUVTexture)
  {
    return;
  }

  auto& upstreamNodeParams = upstreamNode.parameters;
  auto scaleParamIt = upstreamNodeParams.find(_tokens->scale);
  auto biasParamIt = upstreamNodeParams.find(_tokens->bias);

  bool hasScale = (scaleParamIt != upstreamNodeParams.end());
  bool hasBias = (biasParamIt != upstreamNodeParams.end());

  // Bias and scale parameters are missing, for example for J Cube's Maneki asset generated by Multiverse for Maya:
  // https://j-cube.jp/solutions/multiverse/assets
  bool patchParams = (!hasScale && !hasBias);

  if (hasScale && hasBias)
  {
    VtValue boxedScale = scaleParamIt->second;
    VtValue boxedBias = biasParamIt->second;

    if (boxedScale.IsHolding<GfVec4f>() && boxedBias.IsHolding<GfVec4f>())
    {
      GfVec4f scale = boxedScale.UncheckedGet<GfVec4f>();
      GfVec4f bias = boxedBias.UncheckedGet<GfVec4f>();

      // There's a bug with Unity's USD exporter where bias is set to 0 and scale to 1. For example in this asset:
      // https://github.com/usd-wg/assets/blob/25542a54739d36051a4d88a97d3c4e4975238d90/test_assets/AlphaBlendModeTest/AlphaBlendModeTest.usdz
      bool isScale1 = (scale[0] == 1.0f) && (scale[1] == 1.0f) && (scale[2] == 1.0f);
      bool isBias0 = (bias[0] == 0.0f) && (bias[1] == 0.0f) && (bias[2] == 0.0f);

      patchParams = (isScale1 && isBias0);
    }
  }

  if (!patchParams)
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

// Apparently the Unity USD exporter emits (or used to emit) UsdUVTexture nodes with an isSRGB parameter. Found in the wild:
// https://github.com/usd-wg/assets/blob/4c5355bc9bffa96e084961fb5004c829b1c82501/test_assets/AlphaBlendModeTest/AlphaBlendModeTest.usd#L59
// Let's assume that this is part of an older specification version and rename it to "sourceColorSpace".
void _PatchUsdUVTextureIsSrgbParam(HdMaterialNetwork2& network)
{
  for (auto& pathNodePair : network.nodes)
  {
    HdMaterialNode2& node = pathNodePair.second;
    if (node.nodeTypeId != _tokens->ND_UsdUVTexture)
    {
      continue;
    }

    auto& parameters = node.parameters;

    auto isSrgbParam = parameters.find(_tokens->isSRGB);
    if (isSrgbParam == parameters.end())
    {
      continue;
    }

    auto value = isSrgbParam->second;
    TfToken newValue = _tokens->_auto;
    // https://github.com/Unity-Technologies/usd-unity-sdk/blob/307303b25f5fd83e5275a2607b356e43799c38b4/package/com.unity.formats.usd/Dependencies/USD.NET.Unity/Shading/UsdPreviewSurface/TextureReaderSample.cs#L52-L57
    if (value == _tokens->yes)
    {
      newValue = _tokens->sRGB;
    }
    else if (value == _tokens->no)
    {
      newValue = _tokens->raw;
    }

    parameters.erase(isSrgbParam);
    parameters[_tokens->sourceColorSpace] = newValue;
  }
}

void MaterialNetworkPatcher::Patch(HdMaterialNetwork2& network)
{
  _PatchUsdPreviewSurfaceGlossiness(network);

  _PatchUsdPreviewSurfaceSpecular(network);

  if (!getenv(ENVVAR_DISABLE_PATCH_USDPREVIEWSURFACE_NORMALMAP))
  {
    _PatchUsdPreviewSurfaceNormalMap(network);
  }

  _PatchUsdUVTextureIsSrgbParam(network);

  _PatchUsdTypes(network);
}

PXR_NAMESPACE_CLOSE_SCOPE
