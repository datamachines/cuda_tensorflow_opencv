ARG CTO_FROM
FROM ${CTO_FROM}

# Install system packages
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -y \
  && apt-get install -y --no-install-recommends apt-utils \
  && apt-get install -y \ 
    autoconf \
    automake \
    checkinstall \
    cmake \
    curl \
    doxygen \
    file \
    git \
    imagemagick \
    libatk-adaptor \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libblosc-dev \
    libboost-all-dev \
    libbrotli-dev \
    libbz2-dev \
    libcanberra-gtk-module \
    libcharls-dev \
    libdc1394-22-dev \
    libeigen3-dev \
    libfaac-dev \
    libfreetype6-dev \
    libgflags-dev \
    libgif-dev \
    libglew-dev \
    libgoogle-glog-dev \
    libgphoto2-dev \
    libgstreamer-plugins-bad1.0-0 \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    libgtk-3-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    libjpeg-turbo8-dev \
    libjxr-dev \
    liblcms2-dev \
    liblz4-dev \
    liblzma-dev \
    libmp3lame-dev \
    libopenblas-base \
    libopenblas-dev \
    libopencore-amrnb-dev \
    libopencore-amrwb-dev \
    libopenjp2-7-dev \
    libopenjp2-tools \
    libopenmpi-dev \
    libpng-dev \
    libpostproc-dev \
    libprotobuf-dev \
    libpython3-dev \
    libsnappy-dev \
    libsox-dev \
    libsox-fmt-all \ 
    libswscale-dev \
    libtbb-dev \
    libtbb2 \
    libtheora-dev \
    libtool \
    libv4l-dev \
    libvorbis-dev \
    libwebp-dev \
    libx264-dev \
    libxi-dev \
    libxine2-dev \
    libxmu-dev \
    libxvidcore-dev \
    libzmq3-dev \
    libzopfli-dev \
    libzstd-dev \
    pkg-config \
    protobuf-compiler \
    python-imaging-tk \
    python-lxml \
    python-pil \
    python3-tk \
    software-properties-common \
    time \
    unzip \
    v4l-utils \
    x11-apps \
    x264 \
    yasm \
    zip

# Set locale to UTF-8; seems to be ANSI_X3.4-1968 by default (to allow scikit-image to compile)
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && dpkg-reconfigure locales \
    && update-locale LANG=en_US.UTF-8
ENV LANG en_US.UTF-8

# CFLAGS (for imagecodecs)
ENV CFLAGS="-I/usr/include/openjpeg-2.3 -I/usr/include/jxrlib"

# Building libtiff (to allow imagecodecs to compile)
RUN mkdir -p /usr/local/src/libtiff \
  && wget -q --no-check-certificate -c https://gitlab.com/libtiff/libtiff/-/archive/v4.2.0/libtiff-v4.2.0.tar.bz2 -O - | tar --strip-components=1 -xj -C /usr/local/src/libtiff \
  && cd /usr/local/src/libtiff \
  && ./autogen.sh \
  && ./configure \
  && make install \
  && rm -rf /usr/local/src/libtiff

# brunsli (for imagecodecs)
RUN cd /tmp \
  && git clone --depth=1 https://github.com/google/brunsli.git \
  && cd brunsli \
  && git submodule update --init --recursive \
  && cmake -DCMAKE_BUILD_TYPE=Release \
  && make install

# Setup pip
RUN wget -q -O /tmp/get-pip.py --no-check-certificate https://bootstrap.pypa.io/get-pip.py \
  && python3 /tmp/get-pip.py \
  && pip3 install -U pip
RUN pip3 install -U testresources setuptools

# Install Python tools (split install for Nano)
RUN pip3 install -U Cython
RUN pip3 install -U pillow
RUN pip3 install -U matplotlib
RUN pip3 install -U PyWavelets
RUN pip3 install -U kiwisolver
RUN pip3 install -U imagecodecs
RUN pip3 install -U scikit-image
RUN pip3 install -U scikit-learn
RUN pip3 install -U ipython
RUN pip3 install -U notebook
RUN pip3 install -U pandas
RUN pip3 install -U moviepy
RUN pip3 install -U autovizwidget
RUN pip3 install -U jupyter

# Installing PyTorch
# https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-7-0-now-available/72048
# Manual installation for audio and vision components of known releases
RUN cd /tmp \
  && wget -q --no-check-certificate https://nvidia.box.com/shared/static/cs3xn3td6sfgtene6jdvsxlr366m2dhq.whl -O torch-1.7.0-cp36-cp36m-linux_aarch64.whl \
  && pip3 install torch-1.7.0-cp36-cp36m-linux_aarch64.whl \
  && rm -rf /root/.cache/pip torch-1.7.0-cp36-cp36m-linux_aarch64.whl
RUN mkdir -p /usr/local/src/torchvision \
  && wget -q --no-check-certificate https://github.com/pytorch/vision/archive/v0.8.2.tar.gz -O - | tar --strip-components=1 -xz -C /usr/local/src/torchvision \
  && cd /usr/local/src/torchvision \
  && python3 setup.py install  \
  && rm -rf /root/.cache/pip /usr/local/src/torchvision
RUN mkdir -p /usr/local/src/torchaudio \
  && wget -q --no-check-certificate https://github.com/pytorch/audio/archive/v0.7.2.tar.gz -O - | tar --strip-components=1 -xz -C /usr/local/src/torchaudio \
  && cd /usr/local/src/torchaudio \
  && python3 setup.py install  \
  && rm -rf /root/.cache/pip /usr/local/src/torchaudio

# Installing a built-TF compatible keras
ARG CTO_TF_KERAS
RUN pip3 install ${CTO_TF_KERAS} \
  && rm -rf /root/.cache/pip

# Download OpenCV
ARG CTO_OPENCV_VERSION
ARG CTO_NUMPROC=1
ARG CTO_CUDA_BUILD
RUN mkdir -p /usr/local/src/opencv /usr/local/src/opencv_contrib \
  && cd /usr/local/src \
  && wget -q --no-check-certificate https://github.com/opencv/opencv/archive/${CTO_OPENCV_VERSION}.tar.gz -O - | tar --strip-components=1 -xz -C /usr/local/src/opencv \
  && wget -q --no-check-certificate https://github.com/opencv/opencv_contrib/archive/${CTO_OPENCV_VERSION}.tar.gz -O - | tar --strip-components=1 -xz -C /usr/local/src/opencv_contrib \
  && mkdir -p /usr/local/src/opencv/build \
  && cd /usr/local/src/opencv/build \
  && time cmake \
    -DBUILD_DOCS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_opencv_python2=OFF \
    -DBUILD_opencv_python3=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local/ \
    -DCMAKE_INSTALL_TYPE=Release \
    -DENABLE_FAST_MATH=1 \
    -DFORCE_VTK=ON \
    -DINSTALL_C_EXAMPLES=OFF \
    -DINSTALL_PYTHON_EXAMPLES=OFF \
    -DOPENCV_ENABLE_NONFREE=OFF \
    -DOPENCV_EXTRA_MODULES_PATH=/usr/local/src/opencv_contrib/modules \
    -DOPENCV_GENERATE_PKGCONFIG=YES \
    -DWITH_CSTRIPES=ON \
    -DWITH_EIGEN=ON \
    -DWITH_GDAL=ON \
    -DWITH_GSTREAMER=ON \
    -DWITH_GSTREAMER_0_10=OFF \
    -DWITH_GTK=ON \
    -DWITH_IPP=OFF \
    -DWITH_OPENCL=ON \
    -DWITH_OPENMP=ON \
    -DWITH_TBB=ON \
    -DWITH_V4L=ON \
    -DWITH_WEBP=ON \
    -DWITH_XINE=ON \
    ${CTO_CUDA_BUILD} \
    .. \
  && make -j${CTO_NUMPROC} install \
  && sh -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf' \
  && ldconfig \
  && rm -rf /usr/local/src/opencv /usr/local/src/opencv_contrib
## FYI: We are removing the OpenCV directory (in /usr/local/src) to attempt to save additional disk space
# Comment the above line (and remove the \ in the line above) if you want to
#  rerun cmake with additional/modified options AFTER it was built; for example:
# cd /usr/local/src/opencv/build
# cmake -DOPENCV_ENABLE_NONFREE=ON -DBUILD_EXAMPLES=ON -DBUILD_DOCS=ON -DBUILD_TESTS=ON -DBUILD_PERF_TESTS=ON .. && make install

# Add dataframe display widget
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension

RUN touch /.within_container

# https://github.com/opencv/opencv/issues/14884
ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

# Setting up working directory 
RUN mkdir /dmc
WORKDIR /dmc

CMD /bin/bash

LABEL "Author"="Data Machines Corp <help@datamachines.io>"
