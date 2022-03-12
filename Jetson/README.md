# DockerFile for Nvidia Jetson Nano with GPU support for TensorFlow and OpenCV
Revision: 20210218

<!-- vscode-markdown-toc -->
* 1. [Building the images (on a JetsonNano)](#BuildingtheimagesonaJetsonNano)
	* 1.1. [Building a specialized container](#Buildingaspecializedcontainer)
* 2. [A note on AlexyeyAB/darknet](#AnoteonAlexyeyABdarknet)
* 3. [A note on ssh-ing into the Nano and X11 forwarding from within the container](#Anoteonssh-ingintotheNanoandX11forwardingfromwithinthecontainer)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->
<!-- NOTE: joffreykern.markdown-toc use Ctrl+Shift+P to call Generate TOC for MarkDown-->

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

##  1. <a name='BuildingtheimagesonaJetsonNano'></a>Building the images (on a JetsonNano)

Please note that without build caching, on a MAXN-configured Nano with additional swap, each build takes over 3 hours.

The tag for any image built will contain the `datamachines/` organization addition that is found in any of the publicly released pre-built container images.

Use the provided `Makefile` by running `make` to get a list of targets to build:
- `make build_all` will build all container images
- `make jetsonnano-cuda_tensorflow_opencv` will build all the `jetsonnano-cuda_tensorflow_opencv` container images
- use a direct tag to build a specific version; for example `make jetsonnano-cuda_tensorflow_opencv-10.2_2.3_4.5.1`, will build the `datamachines/jetsonnano-cuda_tensorflow_opencv:10.2_2.2_4.5.1-20210218` container image (if such a built is available, see the `Docker Image tag ending` and the list of `Available Docker images to be built` for accurate values).

###  1.1. <a name='Buildingaspecializedcontainer'></a>Building a specialized container

The 20210218 container is based on JetPack 4.5 (L4T R32.5.0), if you need to build a version based based on a different base container, please refer to the tags available at https://ngc.nvidia.com/catalog/containers/nvidia:l4t-tensorflow and reflect this value in the `Makefile`'s `JETPACK_RELEASE` as well as the `STABLE_TF` variables. Just keep in mind that we are not install CUDA or CuDNN but using the ones available within the base container we are pulling.

Note: This base container provided by Nvidia for TensorFlow does include CuDNN, but we are keeping the `cuda` name as we are only providing a limited subset of release.

##  2. <a name='AnoteonAlexyeyABdarknet'></a>A note on AlexyeyAB/darknet

If you follow the steps in the main `README.md` for the project, you will be able to build Darknet to run on the Jetson Nano, after you apply a few changes:
- use the `jetsonnano` version of `cuda_tensorflow_opencv` as your base container,
- the container can not be built with CuDNN, so disable it from the build line,
- the supported architecture needs to be adapted for the Jetson Nano.

Note that the main `README.md` has been updated to use a later version of Yolov4 and this can be repeated here, you only need to adapt the `FROM` line as well as copy the JetsonNano's `perl -i.bak` line to build. 

This reflects as follows:
<pre>
FROM datamachines/jetsonnano-cuda_tensorflow_opencv:10.2_2.3_4.5.1-20210218

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

##  3. <a name='Anoteonssh-ingintotheNanoandX11forwardingfromwithinthecontainer'></a>A note on ssh-ing into the Nano and X11 forwarding from within the container

Reference: [Stackoverflow: Run X application in a Docker container reliably on a server connected via SSH without “--net host”](https://stackoverflow.com/a/48235281)

This is valid not just for headless configuration, but also if your nano is configured to start in `multi-user.target`.

If you want to do X11 displays from within a running container, you will to setup the nano to allow it to accept remote connections to the X11 tunnel (remember to `ssh -X` into your host)

Confirm that you can display X11 after `ssh -X`-ing into your nano by running `xeyes`, if does not work do the step below and retry.

On your nano, edit the `/etc/ssh/sshd_config` file and make sure `ForwardX11` is set to `yes`, and also make sure that `X11UseLocalhost` is set to `no`. Restart `sshd` using `service sshd restart`, logout and ssh back into the nano for it to take effect. Confirm you can still `xeyes`.

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
 
Calling (after `chmod +x`-ing) this script in the directory where your weights file is using `./x11docker.sh` (if you have warning about `ufw` ignore it simply means that you do not have a firewall on your nano, if you see an error about `unable to link authority file` it means that you already have an X11 connection for docker and you might either use the indicated file or reboot the nano to clear the socket) will enable you to simply:
<pre>
./darknet detector demo cfg/coco.data /dmc/yolov4-tiny.cfg /dmc/yolov4-tiny.weights URL
</pre>
to see a live view of the processing of URL (which can be a video file that you are placing in `/dmc` or the url of a live streaming camera). Note that we in this example we are using `tiny` weights (see project main's documentation for further usage, and the github `cfg` directory for the latest configuration), so processing will be slower.
