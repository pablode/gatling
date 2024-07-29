//
// Copyright (C) 2023 Pablo Delgado Krämer
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

#pragma once

#include <pxr/imaging/hd/light.h>

namespace gtl
{
  struct GiScene;
  struct GiSphereLight;
  struct GiDistantLight;
  struct GiRectLight;
  struct GiDiskLight;
  struct GiDomeLight;
}

using namespace gtl;

PXR_NAMESPACE_OPEN_SCOPE

class HdGatlingLight : public HdLight
{
public:
  HdGatlingLight(const SdfPath& id, GiScene* scene);

  HdDirtyBits GetInitialDirtyBitsMask() const override;

protected:
  GfVec3f _CalcBaseEmission(HdSceneDelegate* sceneDelegate, float normalizeFactor);

protected:
  GiScene* _scene;
};

class HdGatlingSphereLight final : public HdGatlingLight
{
public:
  HdGatlingSphereLight(const SdfPath& id, GiScene* scene);

public:
  void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits) override;

  void Finalize(HdRenderParam* renderParam) override;

private:
  GiSphereLight* _giSphereLight;
};


class HdGatlingDistantLight final : public HdGatlingLight
{
public:
  HdGatlingDistantLight(const SdfPath& id, GiScene* scene);

public:
  void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits) override;

  void Finalize(HdRenderParam* renderParam) override;

private:
  GiDistantLight* _giDistantLight;
};


class HdGatlingRectLight final : public HdGatlingLight
{
public:
  HdGatlingRectLight(const SdfPath& id, GiScene* scene);

public:
  void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits) override;

  void Finalize(HdRenderParam* renderParam) override;

private:
  GiRectLight* _giRectLight;
};


class HdGatlingDiskLight final : public HdGatlingLight
{
public:
  HdGatlingDiskLight(const SdfPath& id, GiScene* scene);

public:
  void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits) override;

  void Finalize(HdRenderParam* renderParam) override;

private:
  GiDiskLight* _giDiskLight;
};


class HdGatlingDomeLight final : public HdGatlingLight
{
public:
  HdGatlingDomeLight(const SdfPath& id, GiScene* scene);

public:
  void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits) override;

  void Finalize(HdRenderParam* renderParam) override;

  HdDirtyBits GetInitialDirtyBitsMask() const override;

private:
  void DestroyDomeLight(HdRenderParam* renderParam);

  GiDomeLight* _giDomeLight = nullptr;
};


class HdGatlingSimpleLight final : public HdGatlingLight
{
public:
  HdGatlingSimpleLight(const SdfPath& id, GiScene* scene);

public:
  void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits) override;

  void Finalize(HdRenderParam* renderParam) override;

private:
  GiSphereLight* _giSphereLight = nullptr;
};


PXR_NAMESPACE_CLOSE_SCOPE
