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

struct GiScene;
struct GiSphereLight;
struct GiDistantLight;
struct GiRectLight;
struct GiDomeLight;

PXR_NAMESPACE_OPEN_SCOPE

class HdGatlingLight : public HdLight
{
protected:
  HdGatlingLight(const SdfPath& id, GiScene* scene);

  GfVec3f CalcBaseEmission(HdSceneDelegate* sceneDelegate, float normalizeFactor);

  HdDirtyBits GetInitialDirtyBitsMask() const override;

protected:
  GiScene* m_scene;
};

class HdGatlingSphereLight final : public HdGatlingLight
{
public:
  HdGatlingSphereLight(const SdfPath& id, GiScene* scene);

public:
  void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits) override;

  void Finalize(HdRenderParam* renderParam) override;

private:
  GiSphereLight* m_giSphereLight;
};


class HdGatlingDistantLight final : public HdGatlingLight
{
public:
  HdGatlingDistantLight(const SdfPath& id, GiScene* scene);

public:
  void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits) override;

  void Finalize(HdRenderParam* renderParam) override;

private:
  GiDistantLight* m_giDistantLight;
};


class HdGatlingRectLight final : public HdGatlingLight
{
public:
  HdGatlingRectLight(const SdfPath& id, GiScene* scene);

public:
  void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits) override;

  void Finalize(HdRenderParam* renderParam) override;

private:
  GiRectLight* m_giRectLight;
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
  GiDomeLight* m_giDomeLight = nullptr;
};


class HdGatlingSimpleLight final : public HdGatlingLight
{
public:
  HdGatlingSimpleLight(const SdfPath& id, GiScene* scene);

public:
  void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits) override;

  void Finalize(HdRenderParam* renderParam) override;

private:
  GiSphereLight* m_giSphereLight = nullptr;
  GiDomeLight* m_giDomeLight = nullptr;
};


PXR_NAMESPACE_CLOSE_SCOPE
