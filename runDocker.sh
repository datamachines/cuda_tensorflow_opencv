#!/bin/bash

if [ -z ${CTO+x} ]; then echo "CTO variable not set, unable to continue"; exit 1; fi
if [ "A${CTO}" == "A" ]; then echo "CTO variable empty, unable to continue"; exit 1; fi 

xhost +local:docker
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
nvidia-docker run -it --runtime=nvidia --rm -e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH -v ${PWD}:/dmc --ipc host ${CTO} /bin/bash
xhost -local:docker
