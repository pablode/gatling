
## gatling

<p align="middle">
  <a href="http://pablode.com/cornell_250k.png"><img height=220 src="http://pablode.com/cornell_250k_small.png" /></a>
  <a href="http://pablode.com/salle_100k.png"><img height=220 src="http://pablode.com/salle_100k_small.png" /></a>
</p>
<p align="middle">
  Models downloaded from Morgan McGuire's <a href="https://casual-effects.com/data">Computer Graphics Archive</a>.
</p>

### About

_gatling_ is a GPU path tracer written in C. It's supposed to be a personal research project and learning experience.

Following papers and techniques are implemented:

- Compressed 8-wide BVH building and traversal [\[Ylitie et al. 2017\]](#user-content-ylitie-et-al-2017)
- Spatial splits in BVHs [\[Stich et al. 2009\]](#user-content-stich-et-al-2009)
- Range-based memory scheme for parallel spatial splitting [\[Gobbetti et al. 2016\]](#user-content-gobbetti-et-al-2016)
- SAH with Binning [\[Wald 2007\]](#user-content-wald-2007)
- Wavefront architecture with persistent threads [\[Laine et al. 2013\]](#user-content-laine-et-al-2013)
- Cross-platform memory mapping
- Uniform hemisphere sampling with lambertian BRDF

### Building

CMake 3.14+, a C11 compiler and Vulkan are required.

```
git clone https://github.com/pablode/gatling
mkdir gatling/build && cd gatling/build
cmake .. -Wno-dev -DCMAKE_BUILD_TYPE=Release
cmake --build . --target gp gatling
```

Before path tracing, an intermediate representation of the scene must be built using `gp`, _gatling_'s asset compiler:
```
./bin/gp cornell_box.gltf scene.gsd
```

Make sure there is a camera in the scene and at least one emissive surface. It is recommended to use glTF as the transmission format.  

Next, either disable the system's GPU watchdog or set an appropriate timeout value. For rendering, multiple optional arguments can be provided:
```
./bin/gatling scene.gsd render.png \
    --image-width=1200 \
    --image-height=1200 \
    --spp=256 \
    --max-bounces=4 \
    --rr-bounce-offset=3 \
    --rr-inv-min-term-prob=1.0
```

_gatling_ is optimized for my Pascal GTX 1060 GPU and will most likely not work on old or integrated GPUs.

### Outlook

The general idea is to follow Manuka's _Shade-before-Hit_ architecture with CPU dicing and GPU vertex pre-shading. Universal Scene Description (USD) will be the input format. A texture caching system with support for mip-mapping, compression and tiling is another major goal. [More...](https://github.com/pablode/gatling/projects)

### Further Reading

###### Ylitie et al. 2017
Henri Ylitie, Tero Karras, and Samuli Laine. 2017. Efficient incoherent ray traversal on GPUs through compressed wide BVHs. In Proceedings of High Performance Graphics (HPG ’17). Association for Computing Machinery, New York, NY, USA, Article 4, 1–13. DOI:10.1145/3105762.3105773

###### Stich et al. 2009
Stich, Martin & Friedrich, Heiko & Dietrich, Andreas. 2009. Spatial splits in bounding volume hierarchies. In Proceedings of the Conference on High Performance Graphics 2009 (HPG ’09). Association for Computing Machinery, New York, NY, USA, 7–13. DOI:10.1145/1572769.1572771

###### Gobbetti et al. 2016
Gobbetti, Enrico & Bethe, Wes. Parallel Spatial Splits in Bounding Volume Hierarchies. 2016. Eurographics Symposium on Parallel Graphics and Visualization. DOI:10.2312/pgv.20161179

###### Laine et al. 2013
Samuli Laine, Tero Karras, and Timo Aila. 2013. Megakernels considered harmful: wavefront path tracing on GPUs. In Proceedings of the 5th High-Performance Graphics Conference (HPG '13). Association for Computing Machinery, New York, NY, USA, 137–143. DOI:10.1145/2492045.2492060

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
