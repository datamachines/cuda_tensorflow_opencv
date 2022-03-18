# DockerFile with Nvidia GPU support for TensorFlow and OpenCV
Revision: 20220318

<!-- vscode-markdown-toc -->
* 1. [About](#About)
* 2. [Docker images tag naming](#Dockerimagestagnaming)
* 3. [Building the images](#Buildingtheimages)
* 4. [A note on supported GPU in the Docker Hub builds](#AnoteonsupportedGPUintheDockerHubbuilds)
* 5. [Using the container images](#Usingthecontainerimages)
* 6. [Additional details](#Additionaldetails)
* 7. [Examples of use](#Examplesofuse)
	* 7.1. [Simple OpenCV picture viewer](#SimpleOpenCVpictureviewer)
		* 7.1.1. [Using OpenCV DNN](#UsingOpenCVDNN)
	* 7.2. [Using GPU TensorFlow in your code (only for cudnn- versions)](#UsingGPUTensorFlowinyourcodeonlyforcudnn-versions)
	* 7.3. [Using Jupyter-Notebook (A note on exposing ports)](#UsingJupyter-NotebookAnoteonexposingports)
	* 7.4. [Testing Yolo v4 on your webcam (Linux and GPU only)](#TestingYolov4onyourwebcamLinuxandGPUonly)
		* 7.4.1. [ Darknet Python bindings](#DarknetPythonbindings)
	* 7.5. [Testing PyTorch with CUDA](#TestingPyTorchwithCUDA)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->
<!-- NOTE: joffreykern.markdown-toc use Ctrl+Shift+P to call Generate TOC for MarkDown-->

##  1. <a name='About'></a>About

For TensorFlow GPU, you will need to build the `cudnn_` version.

The base OS for those container images is Ubuntu or DockerHub's `nvidia/cuda` based on Ubuntu. 
More details on the Nvidia base images are available at https://hub.docker.com/r/nvidia/cuda/ . 
In particular, please note that "By downloading these images, you agree to the terms of the license agreements for NVIDIA software included in the images"; with further details on DockerHub version from https://docs.nvidia.com/cuda/eula/index.html#attachment-a

Version history:
- `20191107`: builds a non-CUDA version: `tensorflow_opencv`.
- `20191210`: builds a CuDNN version: `cudnn_tensorflow_opencv`
- `20200211`: making use of Docker 19.03's GPU support and adding information about the OpenCV builds in the `OpenCV_BuildConf` directory.
- `20200327`: added Protobuf, WebP, GStreamer and Eigen to the OpenCV build. 
- `20200423`: added support for OpenCV 3.4.10 and 4.3.0, and added GStreamer plugins to the build. Also added Nvidia Jetson Nano build steps in the `JetsonNano` directory.
- `20200615`: TensorFlow is built from source. Note that TensorFlow will not have GPU support unless it was compiled with CUDNN support. 
- `20200803`: added PyTorch. Removal of `cudnn_` version for CUDA 9.2 with TensorFlow 2.3.0 (minimum needed was 10.1)
- `20201204`: added support for Python 3.7 for TensorFlow 1 builds and Python 3.8 for Tensorflow 2 builds (makes use of the [`deadsnakes/ppa`](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa) and changes the default `python3`). Warning: only `pip3` installed packages will work, not `apt-get` installed ones (Python 3.6 is still the default for Ubuntu 18.04)
- `20210211`: added support for CUDA 11 (using Nvidia's Ubuntu 20.04 based container). CUDA 9.2 and 10.2 versions are still created from Nvidia's Ubuntu 18.04 based containers. 
- `20210414`: updated for OpenCV 3.4.14 and 4.5.2, removed `cuda_` from build target (and built containers)
- `20210420`: moved base container for CPU builds to be based on Ubuntu 20.04 (from 18.04), limiting GPU TF1 builds to CuDNN7, patching TF1 builds (for Python 3.8 and grpc), all containers provide python 3.8 at minimum.
- `20210601`: Added TF 2.5.0 builds for CUDA 11. Providing both CUDA 11.2 and 11.3 builds, as the Nvidia driver needed to run 11.3 is still in beta (465, with 460 as current stable). v2 tag (same release) includes TF2.5.0 with CUDA 10.2
- `20210810`: Added TF 2.5.1 and OpenCV 3.4.15+4.5.3
- `20210812`: Added TF 2.6.0 and updated CUDA to 11.4.1
- `20211027`: Update OpenCV to 3.4.16+4.5.4
- `20211029`: Update min CUDA 11 version to 11.3 (to match PyTorch requirement), and added a README section about testing PyTorch on GPU
- `20211220`: Added TF 2.6.2 and updated PyTorch. Expected to be last release to built TF1 or Ubuntu 18.04 based images.
- `20211222`: Added TF 2.7.0 with CUDA 11 only, which removed Ubuntu 18.04 base images.
- `20220103`: Updated OpenCV to 4.5.5.
- `20220308`: Updated `Jetson` directory (renamed from `JetsonNano`)
- `20220318`: Added TF 2.8.0 and updated PyTorch

`tensorflow_opencv`:
- Builds containers with TensorFlow and OpenCV. Also install, Jupyter, Keras, numpy, pandas, PyTorch and X11 support.
- Can be used on systems without a Nvidia GPU, and the `runDocker.sh` script will setup proper X11 passthrough
- for MacOS X11 passthrough, install the latest XQuartz server and activate the `Security -> Allow connections from network clients` (must logout for it to take effect)

`cudnn_tensorflow_opencv`:
- Builds an Nvidia GPU optimized version of TensorFlow and OpenCV. Also install, Jupyter, Keras, numpy, pandas, PyTorch and X11 support.
- As of the 20200615 version, both OpenCV and TensorFlow are compiled within the container.
- OpenCV integrated additional CUDNN support after October 2019, see [CUDA backend for the DNN module](https://github.com/opencv/opencv/pull/14827).
- For CUDNN, the CUDA backend for DNN module requires CC 5.3 or higher.

`jetson_tensorflow_opencv` (see the `Jetson` directory):
- Builds a Nvidia Jetson `cudnn_tensorflow_opencv` container image based on Nvidia's provided `l4t` containers and adapted from the `Makefile` and `Dockerfile` used for the other builds.

**Docker Images built from this repository are publicly available at https://hub.docker.com/r/datamachines/tensorflow_opencv / https://hub.docker.com/r/datamachines/cudnn_tensorflow_opencv / https://hub.docker.com/r/datamachines/jetson_tensorflow_opencv .**

The [Builds-DockerHub.md](https://github.com/datamachines/cuda_tensorflow_opencv/blob/master/Builds-DockerHub.md) file is a quick way of seeing the list of pre-built container images. When available, a "BuiidInfo" will give the end user a deeper look of the capabilities of said container and installed version. In particular the compiled GPU architecture (see https://en.wikipedia.org/wiki/CUDA#GPUs_supported ).
This is useful for you to decide if you would benefit from re-compiling some container(s) for your specific hardware.

It is possible to use those as `FROM` for your `Dockerfile`; for example: `FROM datamachines/cuda_tensorflow_opencv:10.2_1.15.3_3.4.10-20200615`

##  2. <a name='Dockerimagestagnaming'></a>Docker images tag naming

The image tags follow the `cuda_tensorflow_opencv` naming order.
As such `10.2_1.15.3_3.4.10-20200615` refers to *Cuda 10.2*, *TensorFlow 1.15.3* and *OpenCV 3.4.10*.

Docker images are also tagged with a version information for the date (YYYYMMDD) of the Dockerfile against which they were built from, added at the end of the tag string (following a dash character), such that `cuda_tensorflow_opencv:10.2_1.15.3_3.4.10-20200615` is for the *Dockerfile dating June 15th, 2020*.

Similarly, the `tensorflow_opencv` and `cudnn_tensorflow_opencv` tags follow the same naming convention.

##  3. <a name='Buildingtheimages'></a>Building the images

The tag for any image built will contain the `datamachines/` organization addition that is found in the publicly released pre-built container images.

Use the provided `Makefile` by running `make` to get a list of targets to build:
- `make build_all` will build all container images
- `make tensorflow_opencv` to build all the `tensorflow_opencv` container images
- `make cudnn_tensorflow_opencv` will build all the `cudnn_tensorflow_opencv` container images
- use a direct tag to build a specific version (from the list provided by the call to `make`); for example `make cudnn_tensorflow_opencv-10.2_2.2.0_4.3.0`, will build the `datamachines/cudnn_tensorflow_opencv:10.2_2.2.0_4.3.0-20200615` container image  (if such a built is available, see the `Docker Image tag ending` and the list of `Available Docker images to be built` for accurate values).

The [Builds-DockerHub.md](https://github.com/datamachines/cuda_tensorflow_opencv/blob/master/Builds-DockerHub.md) will give you quick access to the `BuildInfo-OpenCV` and `BuildInfo-TensorFlow` (if available) for a given compilation. Building the image takes time, but we encourage you to modify the `Dockerfile` to reflect your specific needs. If you run a specific `make` you will see the values of the parameters passed to the build, simply set their default `ARG` value to what matches your needs and manually compile, bypassing the `make` by using a form of `docker build --tag="mycto:tag" .` 

##  4. <a name='AnoteonsupportedGPUintheDockerHubbuilds'></a>A note on supported GPU in the Docker Hub builds

In some cases, a minimum nvidia driver version is needed to run specific version of CUDA, [Table 1: CUDA Toolkit and Compatible Driver Versions](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver) and [Table 2: CUDA Toolkit and Minimum Compatible Driver Versions](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) as well as the `nvidia-smi` command on your host will help you determine if a specific version of CUDA will be supported.

It is important to note that not all GPUs are supported in the Docker Hub provided builds. The containers are built for "compute capability (version)" (as defined in the [GPU supported](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) Wikipedia page) of 6.0 and above (ie Pascal and above). 
If you need a different compute capbility, please edit in the `Makefile` the `DNN_ARCH_CUDA` matching the one that you need to build and add your architecture. Then type `make` to see the entire list of containers that the release you have obtained can build and use the exact tag that you want to build to build it locally (on Ubuntu, you will need `docker` and `build-essential` installed at least to do this). For example, from the 20210211 release, you can `make cudnn_tensorflow_opencv-11.2.0_2.4.1_4.5.1`. We can not promise that self built docker image will build or be functional. Building one such container takes a lot of CPU and can take many hours, so we recommend you build only the target you need.

##  5. <a name='Usingthecontainerimages'></a>Using the container images

The use of the provided `runDocker.sh` script present in the source directory allows users to utilize the built image. Dy default, it will set up the X11 passthrough (for Linux and MacOS) and give the user a `/bin/bash` prompt within the running container, as well as mount the calling directory as `/dmc`. A user can test that X11 is functional by using a simple X command such as `xlogo` from the command line.

To use it, the full name of the container image should be passed as the `CONTAINER_ID` environment variable. For example, to use `datamachines/cudnn_tensorflow_opencv-10.2_2.2.0_4.3.0-20200615`, run `CONTAINER_ID=datamachines/cudnn_tensorflow_opencv-10.2_2.2.0_4.3.0-20200615 ./runDocker.sh`. Note that `runDocker.sh` can be called from any location using its full path, so that a user can mount its current working directory as `/dmc` in the running container in order to access local files.

`runDocker.sh` can take multiple arguments; running it without any argument will provide a list of those arguments.

As of Docker 19.03, GPU support is native to the container runtime, as such, we have shifted from the use of `nvidia-docker` to the native `docker [...] --gpus all`. We understand not every user want to use all the GPUs installed on his system, as such, to change this option, change the `D_GPUS` line in the first few lines of `runDocker.sh` to reflect the paramaters that best reflect your system or needs. GPU support is only enabled for the `cuda_` and `cudnn_` images.

Note that the base container runs as root, if you want to run it as a non root user, add `-u $(id -u):$(id -g)` to the `docker` command line but ensure that you have access to the directories you will work in. This can be done using the `-e` command line option of `runDocker.sh`.

##  6. <a name='Additionaldetails'></a>Additional details

- About OpenCV and GPU: In `cuda_tensorflow_opencv` (resp. `cudnn_tensorflow_opencv`), OpenCV is compiled with CUDA (resp. CUDA+CuDNN support), but note that not all of OpenCV's functions are optimized. This is true in particular for some of the `contrib` code.

- A note about `opencv-contrib-python`: The python version of `cv2` built within the container is already built with the "contrib" code (expect the "non free" portion, see the `Makefile` for additional details). `opencv-contrib-python` install another version of `cv2` (as in `import cv2`), as such please be aware that you might lose some of the compiled optimizations.

- Testing GPU availability for TensorFlow: In the `test` directory, you will find a `tf_hw.py` script. You can test it with a `cudnn-` container by adapating the following command:
<pre>
CONTAINER_ID="datamachines/cudnn_tensorflow_opencv:10.2_1.15.3_4.3.0-20200615" ../runDocker.sh -X -N -c python3 -- /dmc/tf_hw.py
</pre>

##  7. <a name='Examplesofuse'></a>Examples of use

###  7.1. <a name='SimpleOpenCVpictureviewer'></a>Simple OpenCV picture viewer

If a user place a picture (named `pic.jpg`) in the directory to be mounted as `/dmc` and the following example script (naming it `display_pic.py3`)

    import numpy as np
    import cv2

    img = cv2.imread('pic.jpg')
    print(img.shape, " ", img.size)
    cv2.imshow('image', img)
    cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()

, adapting `PATH_TO_RUNDOCKER` in `CONTAINER_ID=datamachines/cudnn_tensorflow_opencv-10.2_2.2.0_4.3.0-20200615 PATH_TO_RUNDOCKER/runDocker.sh`, from the provided bash interactive shell, when the user runs `cd /dmc; python3 display_pic.py3`, this will display the picture from the mounted directory on the user's X11 display.

####  7.1.1. <a name='UsingOpenCVDNN'></a>Using OpenCV DNN

This requires a `cudnn_tensorflow_opencv` container and the use of a form of the `--gpus` `docker` options (ex: `docker [...] --gpus all [...]`)

In your `python3` code, make sure to ask OpenCV to use a CUDA backend. This can be achived by adding code similar to:

<pre>
import cv2

net = cv2.dnn.[...]
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
</pre>

You can see more details with this [OpenCV tutorial: YOLO - object detection](https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html?highlight=setpreferablebackend)

We note that other target are available, for example `net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)`

###  7.2. <a name='UsingGPUTensorFlowinyourcodeonlyforcudnn-versions'></a>Using GPU TensorFlow in your code (only for cudnn- versions)

Code written for Tensorflow should follow principles described in https://www.tensorflow.org/guide/using_gpu

In particular, the following section https://www.tensorflow.org/guide/using_gpu#allowing_gpu_memory_growth might be needed to allow proper use of the GPU's memory. In particular:
   
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config, ...)

Note that this often allocates all the GPU memory to one Tensorflow client. If you intend to run multiple Tensorflow containers, limiting the available memory available to the container's Tensorflow can be achieved as described in https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory by instead specifying the percentage of the GPU memory to be used:

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction=0.125
    session = tf.Session(config=config, ...)

The built Docker images do NOT install any models, add/build/download your own in your `Dockerfile` that is `FROM datamachines/cudnn_tensorflow_opencv-10.2_2.2.0_4.3.0-20200615`

For example:

    FROM datamachines/cudnn_tensorflow_opencv-10.2_2.2.0_4.3.0-20200615
    
    # Download tensorflow object detection models
    RUN GIT_SSL_NO_VERIFY=true git clone -q https://github.com/tensorflow/models /usr/local/lib/python3.6/dist-packages/tensorflow/models

    # Install downloaded models
    ENV PYTHONPATH "$PYTHONPATH:/usr/local/lib/python3.6/dist-packages/tensorflow/models/research:/usr/local/lib/python3.6/dist-packages/tensorflow/models/research/slim"
    RUN cd /usr/local/lib/python3.6/dist-packages/tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=.

###  7.3. <a name='UsingJupyter-NotebookAnoteonexposingports'></a>Using Jupyter-Notebook (A note on exposing ports)

By choice, the containers built do not expose any ports, or start any services. This is left to the end-user. To start any, the simpler solution is to base a new container `FROM` one of those containers, expose a port and start said service to be able to access it.

For example, the start and expose Jupyter Notebook (on port `8888`) from the `tensorflow_opencv` container, one could write the following `Dockerfile` and tag it as `jupnb:local`:
<pre>
FROM datamachines/tensorflow_opencv:2.2.0_4.3.0-20200615
EXPOSE 8888
CMD jupyter-notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
</pre>
, using `docker build --tag jupnb:local .`

When starting it using `docker run -p 8888:8888 jupnb:local` to publish the container's port `8888` to the local system's port `8888`, an `http://127.0.0.1:8888/` based URL will shown with the access token.
Using this url in a web browser will grant access to the running instance of Jupyter Notebook.

###  7.4. <a name='TestingYolov4onyourwebcamLinuxandGPUonly'></a>Testing Yolo v4 on your webcam (Linux and GPU only)

It is possible to run Yolov4 using a custom container and building it from source.

In this example we will build [YOLOv4](https://github.com/AlexeyAB/darknet), enabling GPUs (61, 75 and 86 compute), CUDNN, OPENCV, OPENMP, the generation of the `libdarknet.so` which can be used by the `darknet.py` example.

Copy the following lines in a `Dockerfile`
<pre>
FROM datamachines/cudnn_tensorflow_opencv:11.3.1_2.7.0_4.5.5-20220103

RUN mkdir -p /darknet \
    && wget -q --no-check-certificate -c https://github.com/AlexeyAB/darknet/archive/refs/tags/yolov4.tar.gz -O - | tar --strip-components=1 -xz -C /darknet \
    && cd /darknet \
    && perl -i.bak -pe 's%^(GPU|CUDNN|OPENCV|OPENMP|LIBSO)=0%$1=1%g;s%(compute\_61\])%$1 -gencode arch=compute_75,code=[sm_75,compute_75] -gencode arch=compute_86,code=[sm_86,compute_86]%' Makefile \
    && make

WORKDIR /darknet
CMD /bin/bash
</pre>

In the same directory where the `Dockerfile` is, build it using `docker build --tag "cto_darknet:local" .`

Once build is completed, download from https://github.com/AlexeyAB/darknet#pre-trained-models the `cfg-file` and `weights-file` you intend to use, for our examples, we use `yolov4.cfg` and `yolov4.weights`.

From the directory where both files are, run (adapt `RUNDOCKERDIR` with the location of the script):
<pre>
CONTAINER_ID="cto_darknet:local" RUNDOCKERDIR/runDocker.sh -e "--privileged -v /dev/video0:/dev/video0" -c /bin/bash
</pre>
, here we are telling the script to pass to the `docker` command line extra (`-e`) paramaters to run in `privileged` mode (for hardware access) and pass the webcam device (`/dev/video0`) to the container.
By default, this command will also enable X11 display passthrough and mount the current directory (where the cfg and weights are) as `/dmc`.

Because the cfg/weights are accesible in `/dmc` and X11 and webcam can be accessed, running the following command within the newly started container (which started in `/darknet`) will start your webcam (`video0`) and run Yolo v4 on what it sees: 
<pre>
./darknet detector demo cfg/coco.data /dmc/yolov4.cfg /dmc/yolov4.weights
</pre>

For developers, in `/darknet` you will also have the `libdarknet.so` which is needed to use `python3` with `darknet.py` and `darknet_video.py`.

####  7.4.1. <a name='DarknetPythonbindings'></a> Darknet Python bindings

Darknet provides direct python bindings at this point in the form of [darknet_images.py](https://github.com/AlexeyAB/darknet/blob/master/darknet_images.py) and [darknet_video.py](https://github.com/AlexeyAB/darknet/blob/master/darknet_video.py). To test those, you have nothing to do but use the previously built container (`cto_darknet:local`) and run/adapt the following examples (in the default work directory, ie `/darknet`):

`darknet_images.py` example using one of the provided images:
```
python3 darknet_images.py --weights /dmc/yolov4.weights --config_file /dmc/yolov4.cfg --input data/horses.jpg
```

`darknet_video.py` example using webcam:
```
python3 darknet_video.py --weights /dmc/yolov4.weights --config_file /dmc/yolov4.cfg
```

`darknet_video.py` example using video file:
```
python3 darknet_video.py --weights /dmc/yolov4.weights --config_file /dmc/yolov4.cfg --input /dmc/video.mp4
```

Note: the `.py` source code takes additional options, run with `-h` to get the command line help

###  7.5. <a name='TestingPyTorchwithCUDA'></a>Testing PyTorch with CUDA

PyTorch provides examples to test it. Those can be found at https://github.com/pytorch/examples

Here we will test the "Super Resolution" example (for more details, see https://github.com/pytorch/examples/tree/master/super_resolution)

In the directory where the source for `cuda_tensorflow_opencv` is:

```
# First, obtain a copy of the examples
git clone --depth 1 https://github.com/pytorch/examples.git pytorch-examples
# Start a recent container (adapt the CONTAINER_ID for your test), this will mount the current working directory as /dmc
CONTAINER_ID="datamachines/cudnn_tensorflow_opencv:11.3.1_2.6.0_4.5.4-20211029" ./runDocker.sh
# Go into the super resolution example directory
cd pytorch-examples/super_resolution
# Train the model (command line copied from the example README.md, will download the dataset the first time) on GPU (remove the --cuda to use CPU)
python main.py --upscale_factor 3 --batchSize 4 --testBatchSize 100 --nEpochs 30 --lr 0.001 --cuda
# Test the trained super resolver (also copied from example README.md) on GPU
python super_resolve.py --input_image dataset/BSDS300/images/test/16077.jpg --model model_epoch_30.pth --output_filename out.png --cuda
# Note1: If you train on GPU you need to test on GPU: ie make sure to use the --cuda in both command lines
# Note2: You can "time python" to see the speedup from your GPU (using --cuda) versus your CPU (without the --cuda)
```
