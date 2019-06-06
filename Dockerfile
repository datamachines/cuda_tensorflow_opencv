ARG CTO_TENSORFLOW_TAG
FROM tensorflow/tensorflow:$CTO_TENSORFLOW_TAG
# tensorflow-gpu requires nvidia-docker v2 to run
# and is based of nvidia's CUDA Docker image running on Ubuntu 16.04 
#
# Manual Build using (replace <...> with valid values): 
#  docker build \
#    --build-arg CTO_TENSORFLOW_TAG=<tensorflow GPU tag> \
#    --build-arg CTO_CUDA_VERSION=<need to be compatible with tensforlow CUDA version>  \
#    --build-arg CTO_OPENCV_VERSION=<offical release> \
#    --build-arg CTO_NUMPROC=<concurrent number of make ran for compilations> \
#    --tag="cuda_tensorflow_opencv:<should follow the CUDA_TENSORFLOW_OPENCV-RELEASE pattern>" \
#    -f Dockerfile.main \
#    .
#
# Recommended build: follow the options offered by the Makefile
#
# using: CTO_TAG=<valid tag, use docker images to list available tags> ./runDocker.sh


ARG CTO_CUDA_VERSION
ARG CTO_OPENCV_VERSION
ARG CTO_NUMPROC=1

# Install system packages
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update \
  && apt-get install -y --no-install-recommends apt-utils \
  && apt-get install -y --no-install-recommends \ 
    wget unzip vim file \
    build-essential cmake git pkg-config software-properties-common \
    libatlas-base-dev libboost-all-dev \
    x11-apps libgtk2.0-dev libgtk2.0-dev libcanberra-gtk-module libgtk-3-dev qt4-default \
    libtbb2 libtbb-dev \ 
    libjpeg-dev libpng-dev libtiff-dev libpng12-dev imagemagick \
    libv4l-dev libdc1394-22-dev libatk-adaptor \
    python3-dev libpython3-dev python-pil python-lxml python-tk \
    libfreetype6-dev libhdf5-serial-dev libzmq3-dev libjasper-dev \
    libavcodec-dev libavformat-dev libswscale-dev libxvidcore-dev \
    libx264-dev ffmpeg \
    cuda-npp-dev-${CTO_CUDA_VERSION} \
    cuda-cublas-dev-${CTO_CUDA_VERSION} \
    cuda-cufft-dev-${CTO_CUDA_VERSION} \
    cuda-libraries-dev-${CTO_CUDA_VERSION}

# Install core python packages 
RUN wget -q -O /tmp/get-pip.py --no-check-certificate https://bootstrap.pypa.io/get-pip.py \
  && python3 /tmp/get-pip.py \
  && pip install -U pip \
  && pip install -U numpy matplotlib notebook pandas \
    moviepy keras autovizwidget jupyter

# Download & build OpenCV
RUN mkdir -p /usr/local/src \
  && cd /usr/local/src \
  && wget -q --no-check-certificate https://github.com/opencv/opencv/archive/${CTO_OPENCV_VERSION}.tar.gz \
  && tar xfz ${CTO_OPENCV_VERSION}.tar.gz \
  && mv opencv-${CTO_OPENCV_VERSION} opencv \
  && rm ${CTO_OPENCV_VERSION}.tar.gz \
  && wget -q --no-check-certificate https://github.com/opencv/opencv_contrib/archive/${CTO_OPENCV_VERSION}.tar.gz \
  && tar xfz ${CTO_OPENCV_VERSION}.tar.gz \
  && mv opencv_contrib-${CTO_OPENCV_VERSION} opencv_contrib \
  && rm ${CTO_OPENCV_VERSION}.tar.gz
RUN mkdir -p /usr/local/src/opencv/build \
  && cd /usr/local/src/opencv/build \
  && cmake \
    -DCMAKE_INSTALL_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local/ \
    -DOPENCV_GENERATE_PKGCONFIG=ON \
    -DWITH_WEBP=OFF \
    -DINSTALL_C_EXAMPLES=OFF -DINSTALL_PYTHON_EXAMPLES=OFF -DBUILD_EXAMPLES=OFF \
    -DOPENCV_EXTRA_MODULES_PATH=/usr/local/src/opencv_contrib/modules \
    -DBUILD_DOCS=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF \
    -DWITH_TBB=ON -D WITH_EIGEN=ON -D WITH_OPENMP=ON \
    -DWITH_IPP=ON -DWITH_CSTRIPES=ON -DWITH_OPENCL=ON \
    -DWITH_V4L=ON -DENABLE_FAST_MATH=1 -DFORCE_VTK=ON \
    -DWITH_GDAL=ON -DWITH_XINE=ON -DWITH_GTK=ON \
    -DWITH_OPENMP=ON \
    -DWITH_CUDA=ON -DCUDA_FAST_MATH=1 -DWITH_CUBLAS=1 \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-${CTO_CUDA_VERSION} \
    -DCUDA_cublas_LIBRARY=cublas -DCUDA_cufft_LIBRARY=cufft \
    -DCUDA_nppim_LIBRARY=nppim -DCUDA_nppidei_LIBRARY=nppidei \
    -DCUDA_nppif_LIBRARY=nppif -DCUDA_nppig_LIBRARY=nppig \
    -DCUDA_nppim_LIBRARY=nppim -DCUDA_nppist_LIBRARY=nppist \
    -DCUDA_nppisu_LIBRARY=nppisu -DCUDA_nppitc_LIBRARY=nppitc \
    -DCUDA_npps_LIBRARY=npps -DCUDA_nppc_LIBRARY=nppc \
    -DCUDA_nppial_LIBRARY=nppial -DCUDA_nppicc_LIBRARY=nppicc \
    -D CUDA_nppicom_LIBRARY=nppicom \
    .. \
  && make -j${CTO_NUMPROC} install \
  && rm -rf /usr/local/src/opencv

# Add dataframe display widget
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension

# Minimize image size 
RUN (apt-get autoremove -y; apt-get autoclean -y)

# Setting up working directory 
RUN mkdir /dmc
WORKDIR /dmc

CMD bash

# Labels -- kept at the end to maximize layer reuse when possible
LABEL "Author"="Data Machines Corp <help@datamachines.io>"
LABEL description="Preconfigured Ubuntu 16.04 with Nvidia CUDA enabled version of TensorFlow and OpenCV"
