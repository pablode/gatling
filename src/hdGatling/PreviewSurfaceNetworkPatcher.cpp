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

#include "PreviewSurfaceNetworkPatcher.h"

#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/imaging/hd/material.h>
#include <pxr/usd/sdf/assetPath.h>

#include <memory>

const char* ENVVAR_DISABLE_PATCH_USDPREVIEWSURFACE_NORMALMAP = "HDGATLING_MATPATCH_DISABLE_USDPREVIEWSURFACE_NORMALMAP";

PXR_NAMESPACE_OPEN_SCOPE

TF_DEFINE_PRIVATE_TOKENS(
  _tokens,
  (UsdPreviewSurface)
  (UsdUVTexture)
  (UsdPrimvarReader_float)
  (UsdPrimvarReader_float2)
  (UsdPrimvarReader_float3)
  (UsdPrimvarReader_float4)
  (UsdPrimvarReader_int)
  (UsdPrimvarReader_string)
  (UsdPrimvarReader_normal)
  (UsdPrimvarReader_point)
  (UsdPrimvarReader_vector)
  (UsdPrimvarReader_matrix)
  ((_default, "default"))
  (Principled_BSDF)
  (varname)
  (UVmap)
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
  (metallic)
  (roughness)
  (clearcoat)
  (clearcoatRoughness)
  (specular)
  (specularColor)
  (opacity)
  (opacityThreshold)
  (ior)
  (displacement)
  (occlusion)
  (fallback)
  (rgb)
  (r)
  (st)
);

// Following file has 'default' parameters that should be called 'fallback':
// https://usdzshare.com/?ug-gallery=photo-detail&photo_id=490
void _PatchDefaultParam(HdMaterialNetwork2& network)
{
  for (auto& pathNodePair : network.nodes)
  {
    HdMaterialNode2& node = pathNodePair.second;

    if (node.nodeTypeId != _tokens->UsdUVTexture &&
        node.nodeTypeId != _tokens->UsdPrimvarReader_float &&
        node.nodeTypeId != _tokens->UsdPrimvarReader_float2 &&
        node.nodeTypeId != _tokens->UsdPrimvarReader_float3 &&
        node.nodeTypeId != _tokens->UsdPrimvarReader_float4 &&
        node.nodeTypeId != _tokens->UsdPrimvarReader_int &&
        node.nodeTypeId != _tokens->UsdPrimvarReader_string &&
        node.nodeTypeId != _tokens->UsdPrimvarReader_normal &&
        node.nodeTypeId != _tokens->UsdPrimvarReader_point &&
        node.nodeTypeId != _tokens->UsdPrimvarReader_vector &&
        node.nodeTypeId != _tokens->UsdPrimvarReader_matrix)
    {
      continue;
    }

    auto& params = node.parameters;

    auto defaultParam = params.find(_tokens->_default);
    if (defaultParam == params.end())
    {
      continue;
    }

    params[_tokens->fallback] = defaultParam->second;
    params.erase(defaultParam);
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

    if (upstreamNode.nodeTypeId != _tokens->UsdUVTexture)
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

    if (node.nodeTypeId != _tokens->UsdPreviewSurface)
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

    if (node.nodeTypeId != _tokens->UsdPreviewSurface)
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

        if (upstreamNode.nodeTypeId != _tokens->UsdUVTexture)
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

  if (upstreamNode.nodeTypeId != _tokens->UsdUVTexture)
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

// Some Sketchfab assets have a normal parameter of the value (1, 1, 1). For example:
// https://sketchfab.com/3d-models/light-transport-equation-orb-385f55f5d1d34bbc80f91cd86193b78f
// https://sketchfab.com/3d-models/medieval-fantasy-book-06d5a80a04fc4c5ab552759e9a97d91as
void _PatchUsdPreviewSurfaceNormalParamValue(VtValue& value)
{
  if (!value.IsHolding<GfVec3f>())
  {
    return;
  }

  GfVec3f rawVec = value.UncheckedGet<GfVec3f>();
  if (rawVec[0] != 1.0f || rawVec[1] != 1.0f || rawVec[2] != 1.0f)
  {
    return;
  }

  TF_WARN("patching UsdPreviewSurface:normal param value from (1,1,1) to default (0,0,1) (set %s to disable)",
    ENVVAR_DISABLE_PATCH_USDPREVIEWSURFACE_NORMALMAP);

  value = GfVec3f(0, 0, 1);
}

void _PatchUsdPreviewSurfaceFloatInputTypeMismatches(HdMaterialNetwork2& network)
{
  for (auto& pathNodePair : network.nodes)
  {
    HdMaterialNode2& node = pathNodePair.second;
    if (node.nodeTypeId != _tokens->UsdPreviewSurface)
    {
      continue;
    }

    auto& inputs = node.inputConnections;

    // In the Intel Moore Lane 4004 asset, many float inputs of UsdPreviewSurface materials are
    // (such as roughness) were incorrectly connected to the color3 'rgb' output of UsdUVTexture nodes,
    // as opposed to a single-channel output ('r', 'g', 'b'). This is/could have been an artist mistake,
    // or caused by Houdini 19.0/19.5 USD export. https://dpel.aswf.io/4004-moore-lane/
    const auto patchFloatChannelConnection = [&inputs](TfToken inputName)
    {
      auto inputIt = inputs.find(inputName);
      if (inputIt == inputs.end())
      {
        return;
      }

      auto& connections = inputIt->second;
      for (HdMaterialConnection2& connection : connections)
      {
        if (connection.upstreamOutputName != _tokens->rgb)
        {
          continue;
        }

        TF_WARN("patching UsdPreviewSurface:%s input to connect to UsdUVTexture:r", inputName.GetText());

        // We can only guess that e.g. roughness is more likely to be encoded in a separate map
        // as opposed to an occlusion-metalness-roughness map.
        connection.upstreamOutputName = _tokens->r;
      }
    };

    patchFloatChannelConnection(_tokens->metallic);
    patchFloatChannelConnection(_tokens->roughness);
    patchFloatChannelConnection(_tokens->clearcoat);
    patchFloatChannelConnection(_tokens->clearcoatRoughness);
    patchFloatChannelConnection(_tokens->opacity);
    patchFloatChannelConnection(_tokens->opacityThreshold);
    patchFloatChannelConnection(_tokens->ior);
    patchFloatChannelConnection(_tokens->displacement);
    patchFloatChannelConnection(_tokens->occlusion);
  }
}

void _PatchUsdPreviewSurfaceNormalMap(HdMaterialNetwork2& network)
{
  for (auto& pathNodePair : network.nodes)
  {
    HdMaterialNode2& node = pathNodePair.second;
    if (node.nodeTypeId != _tokens->UsdPreviewSurface)
    {
      continue;
    }

    auto& inputs = node.inputConnections;
    auto& parameters = node.parameters;

    if (auto inputIt = inputs.find(_tokens->normal); inputIt != inputs.end())
    {
      auto& connections = inputIt->second;

      for (HdMaterialConnection2& connection : connections)
      {
        _PatchUsdPreviewSurfaceNormalInputConnection(network, connection);
      }
    }

    auto normalParamIt = parameters.find(_tokens->normal);
    if (normalParamIt != parameters.end())
    {
      _PatchUsdPreviewSurfaceNormalParamValue(normalParamIt->second);
    }
  }
}

// Early versions of Blender's USD exporter had a bug where texcoords were written
// to 'st' primvars but UsdUVTexture nodes depended upon 'UVmap' primvars:
// https://projects.blender.org/blender/blender/issues/112407
// To avoid false positives, we only patch primvar readers that are connected to
// texcoord nodes which are connected to preview surfaces of name "Principled_BSDF".
void _PatchUsdPrimvarReaderUVmapVarname(HdMaterialNetwork2& network)
{
  auto handleUsdUVTextureConnection = [&](HdMaterialConnection2& connection)
  {
    HdMaterialNode2& node = network.nodes[connection.upstreamNode];
    if (node.nodeTypeId != _tokens->UsdPrimvarReader_float2)
    {
      return;
    }

    auto& params = node.parameters;

    auto varnameParam = params.find(_tokens->varname);
    if (varnameParam == params.end())
    {
      return;
    }

    VtValue& varnameValue = varnameParam->second;
    if (varnameValue != _tokens->UVmap)
    {
      return;
    }

    varnameValue = _tokens->st;
  };

  auto handleUsdPreviewSurfaceConnection = [&](HdMaterialConnection2& connection)
  {
    HdMaterialNode2& node = network.nodes[connection.upstreamNode];
    if (node.nodeTypeId != _tokens->UsdUVTexture)
    {
      return;
    }

    for (auto& input : node.inputConnections)
    {
      auto& connections = input.second;

      for (HdMaterialConnection2& connection : connections)
      {
        handleUsdUVTextureConnection(connection);
      }
    }
  };

  for (auto& [path, node] : network.nodes)
  {
    if (node.nodeTypeId != _tokens->UsdPreviewSurface)
    {
      continue;
    }

    if (path.GetNameToken() != _tokens->Principled_BSDF)
    {
      continue;
    }

    for (auto& input : node.inputConnections)
    {
      auto& connections = input.second;

      for (HdMaterialConnection2& connection : connections)
      {
        handleUsdPreviewSurfaceConnection(connection);
      }
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
    if (node.nodeTypeId != _tokens->UsdUVTexture)
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

void _PatchUsdTypes(HdMaterialNetwork2& network)
{
  for (auto& pathNodePair : network.nodes)
  {
    HdMaterialNode2& node = pathNodePair.second;

    auto& parameters = node.parameters;

    for (auto& tokenValuePair : parameters)
    {
      VtValue& value = tokenValuePair.second;

#if PXR_VERSION <= 2308
      // Workaround for HdMtlxConvertToString not handling the TfToken type:
      // https://github.com/PixarAnimationStudios/USD/blob/3abc46452b1271df7650e9948fef9f0ce602e3b2/pxr/imaging/hdMtlx/hdMtlx.cpp#L117
      if (value.IsHolding<TfToken>())
      {
        value = value.Cast<std::string>();
      }
#endif

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

void PreviewSurfaceNetworkPatcher::Patch(HdMaterialNetwork2& network)
{
  _PatchDefaultParam(network);

  _PatchUsdPrimvarReaderUVmapVarname(network);

  _PatchUsdPreviewSurfaceGlossiness(network);

  _PatchUsdPreviewSurfaceSpecular(network);

  if (!getenv(ENVVAR_DISABLE_PATCH_USDPREVIEWSURFACE_NORMALMAP))
  {
    _PatchUsdPreviewSurfaceNormalMap(network);
  }

  _PatchUsdPreviewSurfaceFloatInputTypeMismatches(network);

  _PatchUsdUVTextureIsSrgbParam(network);

  _PatchUsdTypes(network);
}

PXR_NAMESPACE_CLOSE_SCOPE
