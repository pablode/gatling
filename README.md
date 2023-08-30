
## gatling

![Evermotion Kitchen](https://github.com/pablode/gatling/assets/3663466/3f4fa9d9-2c40-43a3-96cb-9c490c7a0e8d)

<p align="middle">
  Evermotion <a href="https://evermotion.org/shop/show_product/scene-06-ai48-for-blender/14835">Scene 06 AI48</a>, rendered in Gatling using Blender's Hydra/MaterialX support.
</p>

![Porsche 911 GT3](https://github.com/pablode/gatling/assets/3663466/b6595991-de77-407e-a2e1-af427386382c)

<p align="middle">
  <a href="https://www.artstation.com/marketplace/p/JpgoB/porsche-911-gt3-2022-manthey-racing">Porsche 911 GT3</a>, modified with materials from the <a href="https://matlib.gpuopen.com/main/materials/all">GPUOpen MaterialX library</a>.
</p>

### About

This is my toy path tracer I work on in my free time.

It is exposed as a Hydra render delegate and comes with a standalone that accepts [Universal Scene Description](https://graphics.pixar.com/usd/release/intro.html) (USD) files. It is cross-platform\*, GPU-accelerated, and implements the [MaterialX](https://www.materialx.org/index.html), [NVIDIA MDL](https://www.nvidia.com/en-us/design-visualization/technologies/material-definition-language/) and [UsdPreviewSurface](https://graphics.pixar.com/usd/release/spec_usdpreviewsurface.html) material standards.

Complex BSDFs like [OpenPBR](https://academysoftwarefoundation.github.io/OpenPBR/), Autodesk's Standard Surface, and the glTF shading model are supported using MaterialX and its MDL code generation backend.  The MDL SDK is used to generate evaluation and importance sampling functions as GLSL code, which is compiled to SPIR-V and executed with Vulkan.

\* Hardware ray tracing is required. MacOS will be supported [in the future](https://github.com/KhronosGroup/MoltenVK/issues/427).

### Build

There are [prebuilt binaries](https://github.com/pablode/gatling/releases) which can be copied to the `<USD_INSTALL>/plugin/usd` directory.

Alternatively, for a full source build you need to

- download the <a href="https://developer.nvidia.com/nvidia-mdl-sdk-get-started">MDL SDK</a> (2022.0+) binaries
- download or build <a href="https://github.com/PixarAnimationStudios/USD/tree/v23.11">USD</a> (22.08+) with MaterialX support
- have NASM 2.13+ or YASM 1.2.0+ in your PATH

Do a recursive clone of the repository and set up a build folder:
```
git clone https://github.com/pablode/gatling --recursive
mkdir gatling/build && cd gatling/build
```

Pass following parameters in the CMake generation phase:
```
cmake .. -Wno-dev \
         -DUSD_ROOT=<USD_INSTALL_DIR> \
         -DMDL_ROOT=<MDL_INSTALL_DIR> \
         -DCMAKE_INSTALL_PREFIX=<USD_INSTALL_DIR>/plugin/usd
         -DCMAKE_BUILD_TYPE=Release
```

Build the relevant targets and install the Hydra delegate to the USD plugin folder:
```
cmake --build . --target hdGatling --config Release
cmake --install . --component hdGatling
```

### Usage

Gatling can be used by every application which supports Hydra, either natively or through a plugin.

<p align="middle">
  <img width=740 src="https://github.com/pablode/gatling/assets/3663466/22326db0-3c4d-4913-a68c-371c8b83463a" />
</p>
<p align="middle">
  Alex Treviño's <a href="https://cloud.blender.org/p/gallery/5dd6d7044441651fa3decb56">Junk Shop</a> (<a href="https://creativecommons.org/licenses/by/4.0/">CC BY</a>), distilled to UsdPreviewSurfaces, rendered within <a href="https://openusd.org/release/toolset.html#usdview">usdview</a>.
</p>

A headless standalone is provided that accepts a USD file (.usd, .usda, .usdc, .usdz) as input. It exposes the Hydra render settings as command line arguments:

```
./bin/gatling <scene.usd> render.png \
    --image-width 1200 \
    --image-height 1200 \
    --spp 1024 \
    --max-bounces 8
```

> Note: high sample counts may require adjusting the system watchdog settings.

### Issues

* Features: certain USD prim types (curves, cylinder lights), APIs (UsdLuxShapingAPI, UsdLuxShadowAPI) and features (GeomSubset, subdivision) are not yet supported. UDIM textures, volumes, displacement and other rendering features have yet to be implemented.

* Arbitrary primvar reading: Gatling currently does not implement MDL scene data, which means that MaterialX `geompropvalue` and UsdPreviewSurface `UsdPrimvarReader` nodes are unsupported.

* Real-time editing: changing material parameters, transforming meshes or instances, and adjusting render settings currently result in full or partial cache rebuilds.

### License

Gatling is licensed under the GNU General Public License, as included in the [LICENSE](LICENSE) file.

```

    Copyright (C) 2019 Pablo Delgado Krämer

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <https://www.gnu.org/licenses/>.

```

Licenses of third-party code and libraries are listed in the same file.
