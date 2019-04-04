xhost +local:docker
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
nvidia-docker run -it --runtime=nvidia --rm -e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH -v ${PWD}:/dmc -u $(id -u):$(id -g) --ipc host cuda_tensorflow_opencv:9.0_1.12.0_4.0.1-0.2 /bin/bash
xhost -local:docker
