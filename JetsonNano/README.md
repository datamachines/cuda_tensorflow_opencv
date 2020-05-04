# DockerFile for Nvidia Jetson Nano with GPU support for TensorFlow and OpenCV
Revision: 20200423

The base container for those container images is provided by Nvidia at https://ngc.nvidia.com/catalog/containers/nvidia:l4t-base
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
- `make jetsonnano-cudnn_tensorflow_opencv` will build all the `jetsonnano-cudnn_tensorflow_opencv` container images
- use a direct tag to build a specific version; for example `make jetsonnano-cudnn_tensorflow_opencv-10.0_2.1.0_4.3.0`, will build the `datamachines/jetsonnano-cudnn_tensorflow_opencv:10.0_2.1.0_4.3.0-20200423` container image (if such a built is available, see the `Docker Image tag ending` and the list of `Available Docker images to be built` for accurate values).
