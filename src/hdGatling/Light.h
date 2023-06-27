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


class HdGatlingSphereLight final : public HdLight
{
public:
  HdGatlingSphereLight(GiScene* scene, const SdfPath& id);

public:
  void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits) override;

  void Finalize(HdRenderParam* renderParam) override;

  HdDirtyBits GetInitialDirtyBitsMask() const override;

private:
  GiScene* m_giScene;
  GiSphereLight* m_giSphereLight;
};


class HdGatlingDistantLight final : public HdLight
{
public:
  HdGatlingDistantLight(GiScene* scene, const SdfPath& id);

public:
  void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits) override;

  void Finalize(HdRenderParam* renderParam) override;

  HdDirtyBits GetInitialDirtyBitsMask() const override;

private:
  GiScene* m_giScene;
  GiDistantLight* m_giDistantLight;
};


class HdGatlingRectLight final : public HdLight
{
public:
  HdGatlingRectLight(GiScene* scene, const SdfPath& id);

public:
  void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits) override;

  void Finalize(HdRenderParam* renderParam) override;

  HdDirtyBits GetInitialDirtyBitsMask() const override;

private:
  GiScene* m_giScene;
  GiRectLight* m_giRectLight;
};


class HdGatlingDomeLight final : public HdLight
{
public:
  HdGatlingDomeLight(GiScene* scene, const SdfPath& id);

public:
  void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits) override;

  void Finalize(HdRenderParam* renderParam) override;

  HdDirtyBits GetInitialDirtyBitsMask() const override;

private:
  GiScene* m_giScene;
  GiDomeLight* m_giDomeLight = nullptr;
};


class HdGatlingSimpleLight final : public HdLight
{
public:
  HdGatlingSimpleLight(GiScene* scene, const SdfPath& id);

public:
  void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits) override;

  void Finalize(HdRenderParam* renderParam) override;

  HdDirtyBits GetInitialDirtyBitsMask() const override;

private:
  GiScene* m_giScene;
  GiSphereLight* m_giSphereLight = nullptr;
  GiDomeLight* m_giDomeLight = nullptr;
};


PXR_NAMESPACE_CLOSE_SCOPE
