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

#include "previewSurfaceNetworkPatcher.h"

#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/imaging/hd/material.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/sdf/schema.h>

#include <memory>

PXR_NAMESPACE_OPEN_SCOPE

constexpr static const char* _envvarDisablePatchUsdPreviewSurfaceNormalMap
  = "HDGATLING_MATPATCH_DISABLE_USDPREVIEWSURFACE_NORMALMAP";

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
  (file)
  ((_auto, "auto"))
  (yes)
  (no)
  (sRGB)
  (raw)
  (RAW)
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
  for (auto& [path, node] : network.nodes)
  {
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

    auto paramIt = params.find(_tokens->_default);
    if (paramIt == params.end())
    {
      continue;
    }

    params[_tokens->fallback] = paramIt->second;
    params.erase(paramIt);
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
  auto paramIt = params.find(_tokens->glossiness);
  if (paramIt == params.end())
  {
    return false;
  }

  VtValue value = paramIt->second;
  if (!value.IsHolding<float>())
  {
    return true;
  }

  float glossiness = value.UncheckedGet<float>();
  float roughness = 1.0f - glossiness;

  params[_tokens->roughness] = roughness;
  params.erase(paramIt);
  return true;
}

// Some of Sketchfab's auto-converted assets encode the roughness on the UsdPreviewSurface
// node with a 'glossiness' input. See "Screen Space Reflection Demo: Follmann 2.OG" scene:
// https://sketchfab.com/3d-models/screen-space-reflection-demo-follmann-2og-6164eed28c464c94be8f5268240dc864
void _PatchUsdPreviewSurfaceGlossiness(HdMaterialNetwork2& network)
{
  for (auto& [path, node] : network.nodes)
  {
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
  for (auto& [path, node] : network.nodes)
  {
    if (node.nodeTypeId != _tokens->UsdPreviewSurface)
    {
      continue;
    }

    auto& params = node.parameters;

    // Rename params and change type to Color3
    auto specularParamIt = params.find(_tokens->specular);
    if (specularParamIt != params.end())
    {
      if (specularParamIt->second.IsHolding<float>())
      {
        float specular = specularParamIt->second.UncheckedGet<float>();
        params[_tokens->specularColor] = GfVec3f(specular);
        params.erase(specularParamIt);
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
    _envvarDisablePatchUsdPreviewSurfaceNormalMap);

  upstreamNodeParams[_tokens->scale] = GfVec4f(2.0f, 2.0f, 2.0f, 1.0f);
  upstreamNodeParams[_tokens->bias] = GfVec4f(-1.0f, -1.0f, -1.0f, 0.0f);
}

void _PatchUsdPreviewSurfaceFloatInputTypeMismatches(HdMaterialNetwork2& network)
{
  for (auto& [path, node] : network.nodes)
  {
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
  for (auto& [path, node] : network.nodes)
  {
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

    // Some Sketchfab assets have a normal parameter of the value (1, 1, 1). For example:
    // https://sketchfab.com/3d-models/light-transport-equation-orb-385f55f5d1d34bbc80f91cd86193b78f
    // https://sketchfab.com/3d-models/medieval-fantasy-book-06d5a80a04fc4c5ab552759e9a97d91as
    // Any static value assigned to this input is likely incorrect, so remove it.
    auto normalParamIt = parameters.find(_tokens->normal);
    if (normalParamIt != parameters.end())
    {
      parameters.erase(normalParamIt);
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

    auto varnameParamIt = params.find(_tokens->varname);
    if (varnameParamIt == params.end())
    {
      return;
    }

    VtValue& value = varnameParamIt->second;
    if (value != _tokens->UVmap)
    {
      return;
    }

    value = _tokens->st;
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
  for (auto& [path, node] : network.nodes)
  {
    if (node.nodeTypeId != _tokens->UsdUVTexture)
    {
      continue;
    }

    auto& parameters = node.parameters;

    auto paramIt = parameters.find(_tokens->isSRGB);
    if (paramIt == parameters.end())
    {
      continue;
    }

    auto value = paramIt->second;
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

    parameters.erase(paramIt);
    parameters[_tokens->sourceColorSpace] = newValue;
  }
}

// According to Blender, there are assets which have been authored with uppercase RAW tokens:
// https://projects.blender.org/blender/blender/src/commit/67e3704a450b4c5a01159b80801cc7fdf9e657cb/source/blender/io/usd/intern/usd_reader_material.cc#L1193-L1198
void _PatchUsdUVTextureColorSpaceParam(HdMaterialNetwork2& network)
{
  for (auto& [path, node] : network.nodes)
  {
    if (node.nodeTypeId != _tokens->UsdUVTexture)
    {
      continue;
    }

    auto& parameters = node.parameters;

    auto paramIt = parameters.find(_tokens->sourceColorSpace);
    if (paramIt == parameters.end())
    {
      continue;
    }

    auto& value = paramIt->second;
    if (value == _tokens->RAW)
    {
      value = _tokens->raw;
    }

#if PXR_VERSION >= 2603
    // USD 26.03 renamed 'sourceColorSpace' to 'colorSpace:file'.
    auto colorSpaceParamName = TfToken(SdfPath::JoinIdentifier(SdfFieldKeys->ColorSpace, _tokens->file));
    if (parameters.find(colorSpaceParamName) == parameters.end())
    {
      parameters[colorSpaceParamName] = value;
    }
    parameters.erase(paramIt);
#endif
  }
}

void _PatchUsdTypes(HdMaterialNetwork2& network)
{
  for (auto& [path, node] : network.nodes)
  {
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

  if (!getenv(_envvarDisablePatchUsdPreviewSurfaceNormalMap))
  {
    _PatchUsdPreviewSurfaceNormalMap(network);
  }

  _PatchUsdPreviewSurfaceFloatInputTypeMismatches(network);

  _PatchUsdUVTextureIsSrgbParam(network);

  _PatchUsdUVTextureColorSpaceParam(network);

  _PatchUsdTypes(network);
}

PXR_NAMESPACE_CLOSE_SCOPE

#ifdef HDGATLING_TEST_EXECUTABLE
#include <doctest/doctest.h>

#include <pxr/base/tf/errorMark.h>

#include <string>

PXR_NAMESPACE_OPEN_SCOPE

namespace
{
  HdMaterialNode2& _AddNode(HdMaterialNetwork2& network, const std::string& path, const TfToken& nodeTypeId)
  {
    HdMaterialNode2& node = network.nodes[SdfPath(path)];
    node.nodeTypeId = nodeTypeId;
    return node;
  }

  HdMaterialConnection2 _MakeConnection(const std::string& upstreamNodePath, const TfToken& upstreamOutputName)
  {
    return HdMaterialConnection2{
      .upstreamNode = SdfPath(upstreamNodePath),
      .upstreamOutputName = upstreamOutputName
    };
  }
}

TEST_CASE("PreviewSurfaceNetworkPatcher.DefaultParam")
{
  HdMaterialNetwork2 network;
  TfErrorMark errorMark;

  SUBCASE("Moves default parameter to fallback on supported nodes")
  {
    const TfToken nodeTypes[] = {
      _tokens->UsdUVTexture,
      _tokens->UsdPrimvarReader_float,
      _tokens->UsdPrimvarReader_float2,
      _tokens->UsdPrimvarReader_float3,
      _tokens->UsdPrimvarReader_float4,
      _tokens->UsdPrimvarReader_int,
      _tokens->UsdPrimvarReader_string,
      _tokens->UsdPrimvarReader_normal,
      _tokens->UsdPrimvarReader_point,
      _tokens->UsdPrimvarReader_vector,
      _tokens->UsdPrimvarReader_matrix
    };
    const size_t nodeTypeCount = sizeof(nodeTypes) / sizeof(nodeTypes[0]);

    for (size_t i = 0; i < nodeTypeCount; i++)
    {
      HdMaterialNode2& node = _AddNode(network, "/Node" + std::to_string(i), nodeTypes[i]);
      node.parameters[_tokens->_default] = VtValue(1.0f);
    }

    _PatchDefaultParam(network);

    REQUIRE(network.nodes.size() == nodeTypeCount);
    for (auto& [path, node] : network.nodes)
    {
      CHECK(node.parameters.count(_tokens->_default) == 0);
      REQUIRE(node.parameters.count(_tokens->fallback) == 1);
      CHECK(node.parameters[_tokens->fallback] == VtValue(1.0f));
    }
  }

  SUBCASE("Leaves default parameter on unsupported nodes")
  {
    HdMaterialNode2& node = _AddNode(network, "/Surface", _tokens->UsdPreviewSurface);
    node.parameters[_tokens->_default] = VtValue(1.0f);

    _PatchDefaultParam(network);

    CHECK(node.parameters.count(_tokens->_default) == 1);
    CHECK(node.parameters.count(_tokens->fallback) == 0);
  }

  SUBCASE("Adds no fallback parameter if default is missing")
  {
    HdMaterialNode2& node = _AddNode(network, "/Texture", _tokens->UsdUVTexture);

    _PatchDefaultParam(network);

    CHECK(node.parameters.empty());
  }

  CHECK_MESSAGE(errorMark.IsClean(), "patching posted Tf errors");
}

TEST_CASE("PreviewSurfaceNetworkPatcher.UsdPreviewSurface")
{
  HdMaterialNetwork2 network;
  TfErrorMark errorMark;

  SUBCASE("Converts float glossiness parameter to roughness")
  {
    _AddNode(network, "/Reader", _tokens->UsdPrimvarReader_float);
    HdMaterialNode2& surface = _AddNode(network, "/Surface", _tokens->UsdPreviewSurface);
    surface.parameters[_tokens->glossiness] = VtValue(0.25f);
    surface.inputConnections[_tokens->glossiness].push_back(_MakeConnection("/Reader", TfToken("result")));

    _PatchUsdPreviewSurfaceGlossiness(network);

    CHECK(surface.parameters.count(_tokens->glossiness) == 0);
    REQUIRE(surface.parameters.count(_tokens->roughness) == 1);
    CHECK(surface.parameters[_tokens->roughness] == VtValue(0.75f));
    CHECK(surface.inputConnections.count(_tokens->glossiness) == 1);
    CHECK(surface.inputConnections.count(_tokens->roughness) == 0);
  }

  SUBCASE("Leaves non-float glossiness parameter untouched")
  {
    HdMaterialNode2& surface = _AddNode(network, "/Surface", _tokens->UsdPreviewSurface);
    surface.parameters[_tokens->glossiness] = VtValue(GfVec3f(1.0f));

    _PatchUsdPreviewSurfaceGlossiness(network);

    CHECK(surface.parameters.count(_tokens->roughness) == 0);
    CHECK(surface.parameters[_tokens->glossiness].IsHolding<GfVec3f>());
  }

  SUBCASE("Converts glossiness input to roughness and adds texture scale and bias")
  {
    HdMaterialNode2& texture = _AddNode(network, "/Texture", _tokens->UsdUVTexture);
    HdMaterialNode2& reader = _AddNode(network, "/Reader", _tokens->UsdPrimvarReader_float);
    HdMaterialNode2& surface = _AddNode(network, "/Surface", _tokens->UsdPreviewSurface);
    surface.inputConnections[_tokens->glossiness].push_back(_MakeConnection("/Texture", _tokens->r));
    surface.inputConnections[_tokens->glossiness].push_back(_MakeConnection("/Reader", TfToken("result")));

    _PatchUsdPreviewSurfaceGlossiness(network);

    CHECK(surface.inputConnections.count(_tokens->glossiness) == 0);
    REQUIRE(surface.inputConnections.count(_tokens->roughness) == 1);
    REQUIRE(surface.inputConnections[_tokens->roughness].size() == 2);
    CHECK(surface.inputConnections[_tokens->roughness][0].upstreamNode == SdfPath("/Texture"));
    CHECK(surface.inputConnections[_tokens->roughness][0].upstreamOutputName == _tokens->r);

    REQUIRE(texture.parameters.count(_tokens->scale) == 1);
    CHECK(texture.parameters[_tokens->scale] == VtValue(GfVec4f(-1.0f, -1.0f, -1.0f, -1.0f)));
    REQUIRE(texture.parameters.count(_tokens->bias) == 1);
    CHECK(texture.parameters[_tokens->bias] == VtValue(GfVec4f(1.0f, 1.0f, 1.0f, 0.0f)));

    CHECK(reader.parameters.empty());
  }

  SUBCASE("Keeps existing scale and bias for glossiness input")
  {
    HdMaterialNode2& texture = _AddNode(network, "/Texture", _tokens->UsdUVTexture);
    texture.parameters[_tokens->scale] = VtValue(GfVec4f(0.5f));
    texture.parameters[_tokens->bias] = VtValue(GfVec4f(0.25f));

    HdMaterialNode2& surface = _AddNode(network, "/Surface", _tokens->UsdPreviewSurface);
    surface.inputConnections[_tokens->glossiness].push_back(_MakeConnection("/Texture", _tokens->r));

    _PatchUsdPreviewSurfaceGlossiness(network);

    CHECK(texture.parameters[_tokens->scale] == VtValue(GfVec4f(0.5f)));
    CHECK(texture.parameters[_tokens->bias] == VtValue(GfVec4f(0.25f)));
  }

  SUBCASE("Does nothing if glossiness is absent")
  {
    HdMaterialNode2& surface = _AddNode(network, "/Surface", _tokens->UsdPreviewSurface);

    _PatchUsdPreviewSurfaceGlossiness(network);

    CHECK(surface.parameters.empty());
    CHECK(surface.inputConnections.empty());
  }

  SUBCASE("Converts float specular parameter to specularColor")
  {
    HdMaterialNode2& surface = _AddNode(network, "/Surface", _tokens->UsdPreviewSurface);
    surface.parameters[_tokens->specular] = VtValue(0.5f);

    _PatchUsdPreviewSurfaceSpecular(network);

    CHECK(surface.parameters.count(_tokens->specular) == 0);
    REQUIRE(surface.parameters.count(_tokens->specularColor) == 1);
    CHECK(surface.parameters[_tokens->specularColor] == VtValue(GfVec3f(0.5f)));
  }

  SUBCASE("Leaves non-float specular parameter and input untouched")
  {
    _AddNode(network, "/Texture", _tokens->UsdUVTexture);
    HdMaterialNode2& surface = _AddNode(network, "/Surface", _tokens->UsdPreviewSurface);
    surface.parameters[_tokens->specular] = VtValue(GfVec3f(1.0f, 0.0f, 0.0f));
    surface.inputConnections[_tokens->specular].push_back(_MakeConnection("/Texture", _tokens->r));

    _PatchUsdPreviewSurfaceSpecular(network);

    CHECK(surface.parameters.count(_tokens->specularColor) == 0);
    CHECK(surface.inputConnections.count(_tokens->specular) == 1);
    CHECK(surface.inputConnections.count(_tokens->specularColor) == 0);
  }

  SUBCASE("Renames specular input and texture output")
  {
    _AddNode(network, "/Texture", _tokens->UsdUVTexture);
    _AddNode(network, "/Reader", _tokens->UsdPrimvarReader_float);
    HdMaterialNode2& surface = _AddNode(network, "/Surface", _tokens->UsdPreviewSurface);
    surface.inputConnections[_tokens->specular].push_back(_MakeConnection("/Texture", _tokens->r));
    surface.inputConnections[_tokens->specular].push_back(_MakeConnection("/Reader", TfToken("result")));

    _PatchUsdPreviewSurfaceSpecular(network);

    CHECK(surface.inputConnections.count(_tokens->specular) == 0);
    REQUIRE(surface.inputConnections.count(_tokens->specularColor) == 1);
    REQUIRE(surface.inputConnections[_tokens->specularColor].size() == 2);
    CHECK(surface.inputConnections[_tokens->specularColor][0].upstreamOutputName == _tokens->rgb);
    CHECK(surface.inputConnections[_tokens->specularColor][1].upstreamOutputName == TfToken("result"));
  }

  SUBCASE("Adds scale and bias to normal map textures")
  {
    HdMaterialNode2& texture = _AddNode(network, "/Texture", _tokens->UsdUVTexture);
    HdMaterialNode2& surface = _AddNode(network, "/Surface", _tokens->UsdPreviewSurface);
    surface.inputConnections[_tokens->normal].push_back(_MakeConnection("/Texture", _tokens->rgb));
    surface.parameters[_tokens->normal] = VtValue(GfVec3f(1.0f, 1.0f, 1.0f));

    _PatchUsdPreviewSurfaceNormalMap(network);

    CHECK(texture.parameters[_tokens->scale] == VtValue(GfVec4f(2.0f, 2.0f, 2.0f, 1.0f)));
    CHECK(texture.parameters[_tokens->bias] == VtValue(GfVec4f(-1.0f, -1.0f, -1.0f, 0.0f)));
    CHECK(surface.parameters.count(_tokens->normal) == 0);
  }

  SUBCASE("Replaces identity normal map scale and bias")
  {
    HdMaterialNode2& texture = _AddNode(network, "/Texture", _tokens->UsdUVTexture);
    texture.parameters[_tokens->scale] = VtValue(GfVec4f(1.0f));
    texture.parameters[_tokens->bias] = VtValue(GfVec4f(0.0f));

    HdMaterialNode2& surface = _AddNode(network, "/Surface", _tokens->UsdPreviewSurface);
    surface.inputConnections[_tokens->normal].push_back(_MakeConnection("/Texture", _tokens->rgb));

    _PatchUsdPreviewSurfaceNormalMap(network);

    CHECK(texture.parameters[_tokens->scale] == VtValue(GfVec4f(2.0f, 2.0f, 2.0f, 1.0f)));
    CHECK(texture.parameters[_tokens->bias] == VtValue(GfVec4f(-1.0f, -1.0f, -1.0f, 0.0f)));
  }

  SUBCASE("Keeps non-identity normal map scale and bias")
  {
    HdMaterialNode2& texture = _AddNode(network, "/Texture", _tokens->UsdUVTexture);
    texture.parameters[_tokens->scale] = VtValue(GfVec4f(0.5f));
    texture.parameters[_tokens->bias] = VtValue(GfVec4f(0.25f));

    HdMaterialNode2& surface = _AddNode(network, "/Surface", _tokens->UsdPreviewSurface);
    surface.inputConnections[_tokens->normal].push_back(_MakeConnection("/Texture", _tokens->rgb));

    _PatchUsdPreviewSurfaceNormalMap(network);

    CHECK(texture.parameters[_tokens->scale] == VtValue(GfVec4f(0.5f)));
    CHECK(texture.parameters[_tokens->bias] == VtValue(GfVec4f(0.25f)));
  }

  SUBCASE("Keeps non-vec4 normal map scale and bias")
  {
    HdMaterialNode2& texture = _AddNode(network, "/Texture", _tokens->UsdUVTexture);
    texture.parameters[_tokens->scale] = VtValue(1.0f);
    texture.parameters[_tokens->bias] = VtValue(0.0f);

    HdMaterialNode2& surface = _AddNode(network, "/Surface", _tokens->UsdPreviewSurface);
    surface.inputConnections[_tokens->normal].push_back(_MakeConnection("/Texture", _tokens->rgb));

    _PatchUsdPreviewSurfaceNormalMap(network);

    CHECK(texture.parameters[_tokens->scale] == VtValue(1.0f));
    CHECK(texture.parameters[_tokens->bias] == VtValue(0.0f));
  }

  SUBCASE("Does not patch normal input from non-texture nodes")
  {
    HdMaterialNode2& reader = _AddNode(network, "/Reader", _tokens->UsdPrimvarReader_float3);
    HdMaterialNode2& surface = _AddNode(network, "/Surface", _tokens->UsdPreviewSurface);
    surface.inputConnections[_tokens->normal].push_back(_MakeConnection("/Reader", TfToken("result")));

    _PatchUsdPreviewSurfaceNormalMap(network);

    CHECK(reader.parameters.empty());
  }

  SUBCASE("Connects float inputs to the texture red channel")
  {
    const TfToken floatInputs[] = {
      _tokens->metallic,
      _tokens->roughness,
      _tokens->clearcoat,
      _tokens->clearcoatRoughness,
      _tokens->opacity,
      _tokens->opacityThreshold,
      _tokens->ior,
      _tokens->displacement,
      _tokens->occlusion
    };
    const size_t floatInputCount = sizeof(floatInputs) / sizeof(floatInputs[0]);

    _AddNode(network, "/Texture", _tokens->UsdUVTexture);
    HdMaterialNode2& surface = _AddNode(network, "/Surface", _tokens->UsdPreviewSurface);

    for (size_t i = 0; i < floatInputCount; i++)
    {
      surface.inputConnections[floatInputs[i]].push_back(_MakeConnection("/Texture", _tokens->rgb));
    }
    surface.inputConnections[_tokens->roughness].push_back(_MakeConnection("/Texture", TfToken("g")));

    _PatchUsdPreviewSurfaceFloatInputTypeMismatches(network);

    for (size_t i = 0; i < floatInputCount; i++)
    {
      REQUIRE(surface.inputConnections[floatInputs[i]].size() >= 1);
      CHECK(surface.inputConnections[floatInputs[i]][0].upstreamOutputName == _tokens->r);
    }
    REQUIRE(surface.inputConnections[_tokens->roughness].size() == 2);
    CHECK(surface.inputConnections[_tokens->roughness][1].upstreamOutputName == TfToken("g"));
  }

  SUBCASE("Ignores non-UsdPreviewSurface nodes")
  {
    HdMaterialNode2& node = _AddNode(network, "/Texture", _tokens->UsdUVTexture);
    node.parameters[_tokens->glossiness] = VtValue(0.25f);
    node.parameters[_tokens->specular] = VtValue(0.5f);
    node.parameters[_tokens->normal] = VtValue(GfVec3f(1.0f, 1.0f, 1.0f));
    node.inputConnections[_tokens->roughness].push_back(_MakeConnection("/Texture", _tokens->rgb));

    _PatchUsdPreviewSurfaceGlossiness(network);
    _PatchUsdPreviewSurfaceSpecular(network);
    _PatchUsdPreviewSurfaceNormalMap(network);
    _PatchUsdPreviewSurfaceFloatInputTypeMismatches(network);

    CHECK(node.parameters.count(_tokens->glossiness) == 1);
    CHECK(node.parameters.count(_tokens->specular) == 1);
    CHECK(node.parameters.count(_tokens->normal) == 1);
    CHECK(node.inputConnections[_tokens->roughness][0].upstreamOutputName == _tokens->rgb);
  }

  CHECK_MESSAGE(errorMark.IsClean(), "patching posted Tf errors");
}

TEST_CASE("PreviewSurfaceNetworkPatcher.UsdUVTexture")
{
  HdMaterialNetwork2 network;
  TfErrorMark errorMark;

#if PXR_VERSION >= 2603
  const TfToken colorSpaceFilePath(SdfPath::JoinIdentifier(SdfFieldKeys->ColorSpace, _tokens->file));
#endif

  SUBCASE("Maps isSRGB yes to sRGB")
  {
    HdMaterialNode2& node = _AddNode(network, "/Texture", _tokens->UsdUVTexture);
    node.parameters[_tokens->isSRGB] = VtValue(_tokens->yes);

    _PatchUsdUVTextureIsSrgbParam(network);

    CHECK(node.parameters.count(_tokens->isSRGB) == 0);
    CHECK(node.parameters[_tokens->sourceColorSpace] == VtValue(_tokens->sRGB));
  }

  SUBCASE("Maps isSRGB no to raw")
  {
    HdMaterialNode2& node = _AddNode(network, "/Texture", _tokens->UsdUVTexture);
    node.parameters[_tokens->isSRGB] = VtValue(_tokens->no);

    _PatchUsdUVTextureIsSrgbParam(network);

    CHECK(node.parameters[_tokens->sourceColorSpace] == VtValue(_tokens->raw));
  }

  SUBCASE("Maps other isSRGB values to auto")
  {
    HdMaterialNode2& node = _AddNode(network, "/Texture", _tokens->UsdUVTexture);
    node.parameters[_tokens->isSRGB] = VtValue(_tokens->sRGB);

    _PatchUsdUVTextureIsSrgbParam(network);

    CHECK(node.parameters[_tokens->sourceColorSpace] == VtValue(_tokens->_auto));
  }

  SUBCASE("Lowercases RAW source color space")
  {
    HdMaterialNode2& node = _AddNode(network, "/Texture", _tokens->UsdUVTexture);
    node.parameters[_tokens->sourceColorSpace] = VtValue(_tokens->RAW);

    _PatchUsdUVTextureIsSrgbParam(network); // no isSRGB parameter; must be a no-op
    _PatchUsdUVTextureColorSpaceParam(network);

#if PXR_VERSION >= 2603
    CHECK(node.parameters.count(_tokens->sourceColorSpace) == 0);
    REQUIRE(node.parameters.count(colorSpaceFilePath) == 1);
    CHECK(node.parameters[colorSpaceFilePath] == VtValue(_tokens->raw));
#else
    CHECK(node.parameters[_tokens->sourceColorSpace] == VtValue(_tokens->raw));
#endif
  }

  SUBCASE("Keeps other color spaces")
  {
    HdMaterialNode2& node = _AddNode(network, "/Texture", _tokens->UsdUVTexture);
    node.parameters[_tokens->sourceColorSpace] = VtValue(_tokens->sRGB);

    _PatchUsdUVTextureColorSpaceParam(network);

#if PXR_VERSION >= 2603
    REQUIRE(node.parameters.count(colorSpaceFilePath) == 1);
    CHECK(node.parameters[colorSpaceFilePath] == VtValue(_tokens->sRGB));
#else
    CHECK(node.parameters[_tokens->sourceColorSpace] == VtValue(_tokens->sRGB));
#endif
  }

#if PXR_VERSION >= 2603
  SUBCASE("Does not overwrite existing colorSpace:file")
  {
    HdMaterialNode2& node = _AddNode(network, "/Texture", _tokens->UsdUVTexture);
    node.parameters[_tokens->sourceColorSpace] = VtValue(_tokens->sRGB);
    node.parameters[colorSpaceFilePath] = VtValue(_tokens->raw);

    _PatchUsdUVTextureColorSpaceParam(network);

    CHECK(node.parameters.count(_tokens->sourceColorSpace) == 0);
    CHECK(node.parameters[colorSpaceFilePath] == VtValue(_tokens->raw));
  }
#endif

  SUBCASE("Does nothing without texture parameters")
  {
    HdMaterialNode2& node = _AddNode(network, "/Texture", _tokens->UsdUVTexture);

    _PatchUsdUVTextureIsSrgbParam(network);
    _PatchUsdUVTextureColorSpaceParam(network);

    CHECK(node.parameters.empty());
  }

  SUBCASE("Ignores non-UVTexture nodes")
  {
    HdMaterialNode2& node = _AddNode(network, "/Surface", _tokens->UsdPreviewSurface);
    node.parameters[_tokens->isSRGB] = VtValue(_tokens->yes);
    node.parameters[_tokens->sourceColorSpace] = VtValue(_tokens->RAW);

    _PatchUsdUVTextureIsSrgbParam(network);
    _PatchUsdUVTextureColorSpaceParam(network);

    CHECK(node.parameters[_tokens->isSRGB] == VtValue(_tokens->yes));
    CHECK(node.parameters[_tokens->sourceColorSpace] == VtValue(_tokens->RAW));
  }

  CHECK_MESSAGE(errorMark.IsClean(), "patching posted Tf errors");
}

TEST_CASE("PreviewSurfaceNetworkPatcher.UsdPrimvarReader")
{
  HdMaterialNetwork2 network;
  TfErrorMark errorMark;

  const TfToken diffuseColor("diffuseColor");

  auto addChain = [&network, &diffuseColor](const std::string& surfacePath, const TfToken& readerType, bool addVarname, const TfToken& varname) -> HdMaterialNode2&
  {
    HdMaterialNode2& reader = _AddNode(network, surfacePath + "/Reader", readerType);
    if (addVarname)
    {
      reader.parameters[_tokens->varname] = VtValue(varname);
    }

    HdMaterialNode2& texture = _AddNode(network, surfacePath + "/Texture", _tokens->UsdUVTexture);
    texture.inputConnections[_tokens->st].push_back(_MakeConnection(surfacePath + "/Reader", TfToken("result")));

    HdMaterialNode2& surface = _AddNode(network, surfacePath, _tokens->UsdPreviewSurface);
    surface.inputConnections[diffuseColor].push_back(_MakeConnection(surfacePath + "/Texture", _tokens->rgb));

    return reader;
  };

  SUBCASE("Replaces UVmap varname with st")
  {
    HdMaterialNode2& reader = addChain("/M/Principled_BSDF", _tokens->UsdPrimvarReader_float2, true, _tokens->UVmap);

    _PatchUsdPrimvarReaderUVmapVarname(network);

    REQUIRE(reader.parameters.count(_tokens->varname) == 1);
    CHECK(reader.parameters[_tokens->varname] == VtValue(_tokens->st));
  }

  SUBCASE("Keeps varname if it is not UVmap")
  {
    HdMaterialNode2& reader = addChain("/M/Principled_BSDF", _tokens->UsdPrimvarReader_float2, true, _tokens->st);

    _PatchUsdPrimvarReaderUVmapVarname(network);

    CHECK(reader.parameters[_tokens->varname] == VtValue(_tokens->st));
  }

  SUBCASE("Keeps reader without varname")
  {
    HdMaterialNode2& reader = addChain("/M/Principled_BSDF", _tokens->UsdPrimvarReader_float2, false, TfToken());

    _PatchUsdPrimvarReaderUVmapVarname(network);

    CHECK(reader.parameters.empty());
  }

  SUBCASE("Ignores readers of other types")
  {
    HdMaterialNode2& reader = addChain("/M/Principled_BSDF", _tokens->UsdPrimvarReader_float3, true, _tokens->UVmap);

    _PatchUsdPrimvarReaderUVmapVarname(network);

    CHECK(reader.parameters[_tokens->varname] == VtValue(_tokens->UVmap));
  }

  SUBCASE("Ignores surfaces not named Principled_BSDF")
  {
    HdMaterialNode2& reader = addChain("/M/Surface", _tokens->UsdPrimvarReader_float2, true, _tokens->UVmap);

    _PatchUsdPrimvarReaderUVmapVarname(network);

    CHECK(reader.parameters[_tokens->varname] == VtValue(_tokens->UVmap));
  }

  SUBCASE("Ignores texture inputs without reader")
  {
    HdMaterialNode2& texture = _AddNode(network, "/M/Principled_BSDF/Texture", _tokens->UsdUVTexture);
    texture.inputConnections[_tokens->st].push_back(_MakeConnection("/M/Principled_BSDF/Transform", TfToken("result")));

    HdMaterialNode2& surface = _AddNode(network, "/M/Principled_BSDF", _tokens->UsdPreviewSurface);
    surface.inputConnections[diffuseColor].push_back(_MakeConnection("/M/Principled_BSDF/Texture", _tokens->rgb));

    _PatchUsdPrimvarReaderUVmapVarname(network);

    CHECK(texture.inputConnections[_tokens->st][0].upstreamOutputName == TfToken("result"));
  }

  CHECK_MESSAGE(errorMark.IsClean(), "patching posted Tf errors");
}

TEST_CASE("PreviewSurfaceNetworkPatcher.UsdTypes")
{
  HdMaterialNetwork2 network;
  TfErrorMark errorMark;

  SUBCASE("Replaces asset paths with resolved paths")
  {
    HdMaterialNode2& node = _AddNode(network, "/Texture", _tokens->UsdUVTexture);
    node.parameters[_tokens->file] =
      VtValue(SdfAssetPath("textures/albedo.png", "/resolved/textures/albedo.png"));

    _PatchUsdTypes(network);

    REQUIRE(node.parameters[_tokens->file].IsHolding<std::string>());
    CHECK(node.parameters[_tokens->file].UncheckedGet<std::string>() == "/resolved/textures/albedo.png");
  }

  SUBCASE("Leaves other values untouched")
  {
    HdMaterialNode2& node = _AddNode(network, "/Surface", _tokens->UsdPreviewSurface);
    node.parameters[_tokens->ior] = VtValue(1.5f);
    node.parameters[_tokens->sourceColorSpace] = VtValue(_tokens->sRGB);

    _PatchUsdTypes(network);

    CHECK(node.parameters[_tokens->ior].IsHolding<float>());

#if PXR_VERSION <= 2308
    REQUIRE(node.parameters[_tokens->sourceColorSpace].IsHolding<std::string>());
    CHECK(node.parameters[_tokens->sourceColorSpace].UncheckedGet<std::string>() == "sRGB");
#else
    CHECK(node.parameters[_tokens->sourceColorSpace].IsHolding<TfToken>());
#endif
  }

  CHECK_MESSAGE(errorMark.IsClean(), "patching posted Tf errors");
}

PXR_NAMESPACE_CLOSE_SCOPE
#endif
