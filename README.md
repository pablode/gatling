
## gatling

<p align="middle">
  <a href="http://pablode.com/gatling/pokedstudio.png"><img width=450 src="http://pablode.com/gatling/pokedstudio_sm.png" /></a>
</p>
<p align="middle">
  Pokedstudio's <a href="https://cloud.blender.org/p/gallery/57e51eaa0fcf29412d1f1e76">Blender splash screen</a> <a href="https://creativecommons.org/licenses/by-sa/2.0/legalcode">(CC BY-SA)</a>, modified for and rendered in Gatling.
</p>

### About

This is my GPU path tracer I work on in my free time.  

It features a BVH builder with binned SAH [\[Wald 2007\]](#user-content-wald-2007), spatial splits [\[Stich et al. 2009\]](#user-content-stich-et-al-2009), SAH-preserving widening to nodes of 8 childs, compression, and an efficient traversal kernel [\[Ylitie et al. 2017\]](#user-content-ylitie-et-al-2017).

Gatling is exposed as a Hydra render delegate and comes with a standalone that accepts Universal Scene Description (USD) files [\[Elkoura et al. 2019\]](#user-content-elkoura-et-al-2019).

### Build

You need to build and install version 21.05 of Pixar's <a href="https://github.com/PixarAnimationStudios/USD">USD Framework</a> with MaterialX support enabled.
Additionally, CMake 3.14+, a C11 compiler and the Vulkan SDK are required.

Clone the project and set up a build folder:
```
git clone https://github.com/pablode/gatling
mkdir gatling/build && cd gatling/build
```

Pass the USD install directory in the CMake generation phase:
```
cmake .. -Wno-dev \
         -DUSD_ROOT=<USD_INSTALL_DIR> \
         -DCMAKE_INSTALL_PREFIX=<USD_INSTALL_DIR>/plugin/usd
         -DCMAKE_BUILD_TYPE=Release
```

> Note: If you're using MSVC, be sure to select a 64-bit generator.

Build the relevant targets and install the Hydra delegate to the USD plugin folder:
```
cmake --build . -j8 --target hdGatling gatling --config Release
cmake --install . --component hdGatling
```

### Usage

Gatling can be used by every application which supports Hydra, either natively or through a plugin. It also comes with a headless standalone.
A USD file (.usd, .usda, .usdc, .usdz) is required as input. Make sure there is a camera in the scene and at least one emissive surface.

To play around with Gatling:
```
usdview <scene.usd> --renderer Gatling
```

<p align="middle">
  <a href="http://pablode.com/gatling/usdview_classroom.png"><img width=360 src="http://pablode.com/gatling/usdview_classroom_sm.png" /></a>
</p>
<p align="middle">
  Christophe Seux's <a href="https://www.blender.org/download/demo-files/">Class room</a>
  rendered using Gatling inside <a href="https://graphics.pixar.com/usd/docs/USD-Toolset.html#USDToolset-usdview">Pixar's usdview</a> tool.
</p>

To output an actual image:
```
./bin/gatling <scene.usd> render.png \
    --image-width 1200 \
    --image-height 1200 \
    --spp 1024 \
    --max-bounces 8
```

> Note: Disable the system's GPU watchdog or set an appropriate timeout value.

### Outlook

The idea is to follow Manuka's _Shade-before-Hit_ architecture with CPU dicing and GPU vertex pre-shading. Material networks are going to be translated to GLSL or SPIR-V shader code. [More...](https://github.com/pablode/gatling/projects)

### Further Reading

###### Elkoura et al. 2019
George Elkoura, Sebastian Grassia, Sunya Boonyatera, Alex Mohr, Pol Jeremias-Vila, and Matt Kuruc. 2019. A deep dive into universal scene description and hydra. In ACM SIGGRAPH 2019 Courses (SIGGRAPH '19). Association for Computing Machinery, New York, NY, USA, Article 1, 1–48. DOI:10.1145/3305366.3328033

###### Ylitie et al. 2017
Henri Ylitie, Tero Karras, and Samuli Laine. 2017. Efficient incoherent ray traversal on GPUs through compressed wide BVHs. In Proceedings of High Performance Graphics (HPG ’17). Association for Computing Machinery, New York, NY, USA, Article 4, 1–13. DOI:10.1145/3105762.3105773

###### Stich et al. 2009
Stich, Martin & Friedrich, Heiko & Dietrich, Andreas. 2009. Spatial splits in bounding volume hierarchies. In Proceedings of the Conference on High Performance Graphics 2009 (HPG ’09). Association for Computing Machinery, New York, NY, USA, 7–13. DOI:10.1145/1572769.1572771

###### Wald 2007
Ingo Wald. 2007. On fast Construction of SAH-based Bounding Volume Hierarchies. In Proceedings of the 2007 IEEE Symposium on Interactive Ray Tracing (RT '07). IEEE Computer Society, USA, 33–40. DOI:10.1109/RT.2007.4342588

### License

```

    Copyright (C) 2020 Pablo Delgado Krämer

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
