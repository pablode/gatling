
## gatling

<p align="middle">
  <a href="http://pablode.com/gatling/wwc_mustang.png"><img width=460 src="http://pablode.com/gatling/wwc_mustang_sm.png" /></a>
</p>
<p align="middle">
  1965 Ford Mustang Fastback from <a href="https://wirewheelsclub.com/models/1965-ford-mustang-fastback/">Wire Wheels Club</a>, rendered in Gatling.
</p>

### About

This is my toy path tracer I work on in my free time.

It is exposed as a Hydra render delegate and comes with a standalone that accepts Universal Scene Description (USD) files [\[Elkoura et al. 2019\]](#user-content-elkoura-et-al-2019). Both UsdPreviewSurface and MaterialX [\[Smythe et al. 2019\]](#user-content-smythe-et-al-2019) material network nodes are supported.

Gatling features a BVH builder with binned SAH [\[Wald 2007\]](#user-content-wald-2007), spatial splits [\[Stich et al. 2009\]](#user-content-stich-et-al-2009), SAH-preserving widening to nodes of 8 childs, compression, and an efficient traversal kernel [\[Ylitie et al. 2017\]](#user-content-ylitie-et-al-2017). Complex BSDFs like the Autodesk Standard Surface and the Disney BRDF are supported and importance sampled.

### Build

You need to

- install the <a href="https://vulkan.lunarg.com/">Vulkan SDK</a>
- download the <a href="https://developer.nvidia.com/nvidia-mdl-sdk-get-started">MDL SDK 2021.1.2</a> binaries
- build the <a href="https://github.com/PixarAnimationStudios/USD/tree/dev">USD dev</a> branch with MaterialX support
- have NASM 2.13+ or YASM 1.2.0+ in your PATH

> Note: EXR export requires building USD with OpenImageIO support.

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

> Note: If you're using MSVC, be sure to select a 64-bit generator.

Build the relevant targets and install the Hydra delegate to the USD plugin folder:
```
cmake --build . -j8 --target hdGatling gatling --config Release
cmake --install . --component hdGatling
```

#### DXC

On Windows, Microsoft's DirectX Shader Compiler (DXC) can be used instead of Khronos's glslang. This allows for validation of the generated HLSL shader.

Download the <a href="https://github.com/microsoft/DirectXShaderCompiler/releases/tag/v1.6.2106">DXC June 2021 binaries</a> and set `-DDXC_ROOT` in the CMake generation phase to point to the unpacked folder. You can switch between both shader compilers using the `-DGATLING_USE_DXC` option.

### Usage

Gatling can be used by every application which supports Hydra, either natively or through a plugin.

<p align="middle">
  <a href="http://pablode.com/gatling/usdview_coffeemaker.png"><img width=360 src="http://pablode.com/gatling/usdview_coffeemaker_sm.png" /></a>
</p>
<p align="middle">
  <a href="https://www.blendswap.com/blend/16368">cekuhnen's Coffee Maker</a> (<a href="https://creativecommons.org/licenses/by/2.0/legalcode">CC-BY</a>), slightly modified, rendered using Gatling inside <a href="https://graphics.pixar.com/usd/docs/USD-Toolset.html#USDToolset-usdview">Pixar's usdview</a> tool.
</p>

A headless standalone is provided that accepts a USD file (.usd, .usda, .usdc, .usdz) as input. Make sure that there is a polygonal light source in the scene.

```
./bin/gatling <scene.usd> render.png \
    --image-width 1200 \
    --image-height 1200 \
    --spp 1024 \
    --max-bounces 8
```

> Note: Disable the system's GPU watchdog or set an appropriate timeout value.

### Outlook

Basic texturing is the next important feature. After that, support for MDL materials will be considered.

### Further Reading

###### Smythe et al. 2019
Doug Smythe, Jonathan Stone, Davide Pesare, Henrik Edström. 2019. MaterialX: An Open Standard for Network-Based CG Object Looks. ASWF Open Source Day SIGGRAPH 2019. Retrieved October 25, 2021 from https://www.materialx.org/assets/MaterialX_Sig2019_BOF_slides.pdf.

###### Elkoura et al. 2019
George Elkoura, Sebastian Grassia, Sunya Boonyatera, Alex Mohr, Pol Jeremias-Vila, and Matt Kuruc. 2019. A deep dive into universal scene description and hydra. In ACM SIGGRAPH 2019 Courses (SIGGRAPH '19). Association for Computing Machinery, New York, NY, USA, Article 1, 1–48. DOI:10.1145/3305366.3328033

###### Ylitie et al. 2017
Henri Ylitie, Tero Karras, and Samuli Laine. 2017. Efficient incoherent ray traversal on GPUs through compressed wide BVHs. In Proceedings of High Performance Graphics (HPG ’17). Association for Computing Machinery, New York, NY, USA, Article 4, 1–13. DOI:10.1145/3105762.3105773

###### Stich et al. 2009
Stich, Martin & Friedrich, Heiko & Dietrich, Andreas. 2009. Spatial splits in bounding volume hierarchies. In Proceedings of the Conference on High Performance Graphics 2009 (HPG ’09). Association for Computing Machinery, New York, NY, USA, 7–13. DOI:10.1145/1572769.1572771

###### Wald 2007
Ingo Wald. 2007. On fast Construction of SAH-based Bounding Volume Hierarchies. In Proceedings of the 2007 IEEE Symposium on Interactive Ray Tracing (RT '07). IEEE Computer Society, USA, 33–40. DOI:10.1109/RT.2007.4342588

### License

Gatling is licensed under the GNU General Public License, as included in the [LICENSE](LICENSE) file.

```

    Copyright (C) 2019-2022 Pablo Delgado Krämer

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

It contains code from the Ray Tracing Gems I book, which is [MIT](docs/licenses/LICENSE.MIT.rtgems) licensed and copyrighted:

* Copyright 2019 NVIDIA Corporation

It contains code from the MDL SDK, which is [BSD](docs/licenses/LICENSE.BSD-3.mdl-sdk) licensed and copyrighted:

* Copyright (c) 2018-2021, NVIDIA CORPORATION. All rights reserved.
