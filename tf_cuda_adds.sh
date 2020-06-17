#!/bin/bash

if [ "A$1" == "A10.0" ]; then
    cd /usr/local/nvidia && mkdir lib
    cd /tmp && apt-get download cuda-cusparse-10-0 && mkdir _tmp && dpkg -x cuda-cusparse-10-0*.deb _tmp && rsync -L _tmp/usr/local/cuda-10.0/lib64/libcusparse.so.10.0 /usr/local/nvidia/lib/. && rm -rf _tmp cuda-*.deb
    x="cudart" && cd /tmp && apt-get download cuda-${x}-10-0 && mkdir $x && dpkg -x cuda-$x-10-0*.deb $x && rsync -L $x/usr/local/cuda-10.0/lib64/lib$x.so.10.0 /usr/local/nvidia/lib/. && rm -rf $x cuda-*.deb
    x="cufft" && cd /tmp && apt-get download cuda-${x}-10-0 && mkdir $x && dpkg -x cuda-$x-10-0*.deb $x && rsync -L $x/usr/local/cuda-10.0/lib64/lib$x.so.10.0 /usr/local/nvidia/lib/. && rm -rf $x cuda-*.deb
    x="curand" && cd /tmp && apt-get download cuda-${x}-10-0 && mkdir $x && dpkg -x cuda-$x-10-0*.deb $x && rsync -L $x/usr/local/cuda-10.0/lib64/lib$x.so.10.0 /usr/local/nvidia/lib/. && rm -rf $x cuda-*.deb
    x="cusolver" && cd /tmp && apt-get download cuda-${x}-10-0 && mkdir $x && dpkg -x cuda-$x-10-0*.deb $x && rsync -L $x/usr/local/cuda-10.0/lib64/lib$x.so.10.0 /usr/local/nvidia/lib/. && rm -rf $x cuda-*.deb
    x="cusparse" && cd /tmp && apt-get download cuda-${x}-10-0 && mkdir $x && dpkg -x cuda-$x-10-0*.deb $x && rsync -L $x/usr/local/cuda-10.0/lib64/lib$x.so.10.0 /usr/local/nvidia/lib/. && rm -rf $x cuda-*.deb
    cd /usr/local/nvidia/lib && ln -s ../../cuda-10.0/lib64/libcublas.so libcublas.so.10.0
fi

exit 0
