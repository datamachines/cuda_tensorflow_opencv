# DockerFile for Nvidia Jetson Nano with GPU support for TensorFlow and OpenCV
Revision: 20210218

20210218: Changed the base image from https://ngc.nvidia.com/catalog/containers/nvidia:l4t-base to https://ngc.nvidia.com/catalog/containers/nvidia:l4t-tensorflow
Important: the base image is based on JetPack 4.5 (L4T R32.5.0), if this is not the JetPack version that you are using, please see "Building the images"

Please refer to the following for further details https://github.com/NVIDIA/nvidia-docker/wiki/NVIDIA-Container-Runtime-on-Jetson
Because the `L4T BSP EULA` includes redistribution rights, we are able provide pre-compiled builds.
In particular, please note that "By downloading these images, you agree to the terms of the license agreements for NVIDIA software included in the images"

Publicly available builds can be found at https://hub.docker.com/r/datamachines/jetsonnano-cuda_tensorflow_opencv

Most of the `README.md` in the parent directory explains the logic behind this tool, including the changes to said versions, such as:
- Docker images tag naming
- Using the container images
- Making use of the container

## Building the images (on a JetsonNano)

Please note that without build caching, on a MAXN-configured Nano with additional swap, each build takes over 3 hours.

The tag for any image built will contain the `datamachines/` organization addition that is found in any of the publicly released pre-built container images.

Use the provided `Makefile` by running `make` to get a list of targets to build:
- `make build_all` will build all container images
- `make jetsonnano-cuda_tensorflow_opencv` will build all the `jetsonnano-cuda_tensorflow_opencv` container images
- use a direct tag to build a specific version; for example `make jetsonnano-cuda_tensorflow_opencv-10.2_2.3_4.5.1`, will build the `datamachines/jetsonnano-cuda_tensorflow_opencv:10.2_2.2_4.5.1-20210218` container image (if such a built is available, see the `Docker Image tag ending` and the list of `Available Docker images to be built` for accurate values).

### Building a specialized container

The 20210218 container is based on JetPack 4.5 (L4T R32.5.0), if you need to build a version based based on a different base container, please refer to the tags available at https://ngc.nvidia.com/catalog/containers/nvidia:l4t-tensorflow and reflect this value in the `Makefile`'s `JETPACK_RELEASE` as well as the `STABLE_TF` variables. Just keep in mind that we are not install CUDA or CuDNN but using the ones available within the base container we are pulling.

Note: This base container provided by Nvidia for TensorFlow does include CuDNN, but we are keeping the `cuda` name as we are only providing a limited subset of release.

## A note on AlexyeyAB/darknet

If you follow the steps in the main `README.md` for the project, you will be able to build Darknet to run on the Jetson Nano, after you apply a few changes:
- use the `jetsonnano` version of `cuda_tensorflow_opencv` as your base container,
- the container can not be built with CuDNN, so disable it from the build line,
- the supported architecture needs to be adapted for the Jetson Nano.

This reflects as follows:
<pre>
FROM datamachines/jetsonnano-cuda_tensorflow_opencv:10.0_2.1.0_4.3.0-20200515

RUN mkdir -p /wrk/darknet \
    && cd /wrk \
    && wget -q -c https://github.com/AlexeyAB/darknet/archive/darknet_yolo_v4_pre.tar.gz -O - | tar --strip-components=1 -xz -C /wrk/darknet \
    && cd darknet \
    && perl -i.bak -pe 's%^(GPU|OPENCV|OPENMP|LIBSO)=0%$1=1%g;s%^\#\s*(ARCH=.+compute\_53\])$%$1%' Makefile \
    && make

WORKDIR /wrk/darknet
CMD /bin/bash
</pre>

You can then refer to the project's main documentation for further usage details.
