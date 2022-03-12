# DockerFile for Nvidia Jetson with GPU support for TensorFlow and OpenCV
Revision: 20220308

<!-- vscode-markdown-toc -->
* 1. [Building the images (on a Jetson)](#BuildingtheimagesonaJetson)
	* 1.1. [Tag naming convention](#Tagnamingconvention)
	* 1.2. [Building a specialized container](#Buildingaspecializedcontainer)
* 2. [A note on AlexyeyAB/darknet](#AnoteonAlexyeyABdarknet)
* 3. [A note on ssh-ing into the Jetson and X11 forwarding from within the container](#Anoteonssh-ingintotheJetsonandX11forwardingfromwithinthecontainer)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->
<!-- NOTE: joffreykern.markdown-toc use Ctrl+Shift+P to call Generate TOC for MarkDown-->

Changelog:
- 20210218: Changed the base image from https://ngc.nvidia.com/catalog/containers/nvidia:l4t-base to https://ngc.nvidia.com/catalog/containers/nvidia:l4t-tensorflow
Important: the base image is based on JetPack 4.5 (L4T R32.5.0), if this is not the JetPack version that you are using, please see "Building the images"
- 20220308: Making use of JetPack 4.6 (L4T R32.6.1) base image, and building for 5.3, 6.2 and 7.2 to support more than just the Jetson Nano

Please refer to the following for further details https://github.com/NVIDIA/nvidia-docker/wiki/NVIDIA-Container-Runtime-on-Jetson
Because the `L4T BSP EULA` includes redistribution rights, we are able provide pre-compiled builds.
In particular, please note that "By downloading these images, you agree to the terms of the license agreements for NVIDIA software included in the images"

Publicly available builds can be found at https://hub.docker.com/r/datamachines/jetson_tensorflow_opencv

Most of the `README.md` in the parent directory explains the logic behind this tool, including the changes to said versions, such as:
- Docker images tag naming
- Using the container images
- Making use of the container

##  1. <a name='BuildingtheimagesonaJetson'></a>Building the images (on a Jetson)

Currently built on a Jetson Nano, please note that without build caching, on a MAXN-configured Nano with additional swap, each build takes over 3 hours.

Important: the default runtime for `docker` must be `nvidia` for build to work.

The tag for any image built will contain the `datamachines/` organization addition that is found in any of the publicly released pre-built container images.

Use the provided `Makefile` by running `make` to get a list of targets to build:
- `make build_all` will build all container images
- `make jetson_tensorflow_opencv` will build all the `jetson_tensorflow_opencv` target container images
- use a direct tag to build a specific version; for example `make jetson_tensorflow_opencv-r32.6.1_2.5_4.5.5`, will build the `datamachines/jetson_tensorflow_opencv:r32.6.1_2.5_4.5.5-20220308` container image.

###  1.1. <a name='Tagnamingconvention'></a>Tag naming convention

`jetson_tensorflow_opencv` follows the name of its `_` separated components: Jetson JetPack release number, Tensorflow version, OpenCV version, followed by the revision date of this batch.

For example, `datamachines/jetson_tensorflow_opencv:r32.6.1_2.5_4.5.5-20220308` is JetPack R32.6.1, Tensorflow 2.5 and OpenCV 4.5.5, from the 2022-03-08 revision.

###  1.2. <a name='Buildingaspecializedcontainer'></a>Building a specialized container

The 20220308 container is based on JetPack 4.6 (L4T R32.6.1), if you need to build a version based based on a different base container, please refer to the tags available at https://ngc.nvidia.com/catalog/containers/nvidia:l4t-tensorflow and reflect this value in the `Makefile`'s `JETPACK_RELEASE` as well as the `STABLE_TF` variables. Just keep in mind that we are not install CUDA or CuDNN but using the ones available within the base container we are pulling.

Note: This base container provided by Nvidia for TensorFlow does include CuDNN, but we are keeping the `cuda` name as we are only providing a limited subset of release.

##  2. <a name='AnoteonAlexyeyABdarknet'></a>A note on AlexyeyAB/darknet

If you follow the steps in the main `README.md` for the project, you will be able to build Darknet to run on the Jetson, after you apply a few changes:
- use the `jetson` version of `cudnn_tensorflow_opencv` as your base container (or `tensorflow_opencv`),
- the container can not be built with CuDNN, so disable it from the build line,
- the supported architecture needs to be adapted for the Jetson.
- Note that not all software incorporated in the `tensorflow_opencv` versions might be on all releases, you are invited to check the `BuildConf` directories for details

Note that the main `README.md` has been updated to use a later version of Yolov4 and this can be repeated here, you only need to adapt the `FROM` line as well as copy the Jetson specific's `perl -i.bak` line to build (adapt to your architecture: 5.3, 6.2 or 7.2).

This reflects as follows:
<pre>
FROM datamachines/jetson_tensorflow_opencv:r32.6.1_2.5_4.5.5-20220308

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

##  3. <a name='Anoteonssh-ingintotheJetsonandX11forwardingfromwithinthecontainer'></a>A note on ssh-ing into the Jetson and X11 forwarding from within the container

Reference: [Stackoverflow: Run X application in a Docker container reliably on a server connected via SSH without “--net host”](https://stackoverflow.com/a/48235281)

This is valid not just for headless configuration, but also if your Jetson is configured to start in `multi-user.target`.

If you want to do X11 displays from within a running container, you will to setup the Jetson to allow it to accept remote connections to the X11 tunnel (remember to `ssh -X` into your host)

Confirm that you can display X11 after `ssh -X`-ing into your Jetson by running `xeyes`, if does not work do the step below and retry.

On your Jetson, edit the `/etc/ssh/sshd_config` file and make sure `ForwardX11` is set to `yes`, and also make sure that `X11UseLocalhost` is set to `no`. Restart `sshd` using `service sshd restart`, logout and ssh back into the Jetson for it to take effect. Confirm you can still `xeyes`.

Next, adapt the `docker` line of this `x11docker.sh` script (her we call the `cto_darknet:local` container built in the above section)
<pre>
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | sudo xauth -f $XAUTH nmerge -
sudo chmod 777 $XAUTH
X11PORT=`echo $DISPLAY | sed 's/^[^:]*:\([^\.]\+\).*/\1/'`
TCPPORT=`expr 6000 + $X11PORT`
sudo ufw allow from 172.17.0.0/16 to any port $TCPPORT proto tcp 
DISPLAY=`echo $DISPLAY | sed 's/^[^:]*\(.*\)/172.17.0.1\1/'`
docker run -ti --rm -e DISPLAY=$DISPLAY -v $XAUTH:$XAUTH \
   -e XAUTHORITY=$XAUTH -v `pwd`:/dmc cto_darknet:local 
</pre>
 
Calling (after `chmod +x`-ing) this script in the directory where your weights file is using `./x11docker.sh` (if you have warning about `ufw` ignore it simply means that you do not have a firewall on your Jetson, if you see an error about `unable to link authority file` it means that you already have an X11 connection for docker and you might either use the indicated file or reboot the Jetson to clear the socket) will enable you to simply:
<pre>
./darknet detector demo cfg/coco.data /dmc/yolov4-tiny.cfg /dmc/yolov4-tiny.weights URL
</pre>
to see a live view of the processing of URL (which can be a video file that you are placing in `/dmc` or the url of a live streaming camera). Note that we in this example we are using `tiny` weights (see project main's documentation for further usage, and the github `cfg` directory for the latest configuration), so processing will be slower.
