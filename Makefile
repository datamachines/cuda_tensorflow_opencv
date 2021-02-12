# Needed SHELL since I'm using zsh
SHELL := /bin/bash
.PHONY: all build_all actual_build build_prep

# Release to match data of Dockerfile and follow YYYYMMDD pattern
CTO_RELEASE=20210211

# Maximize build speed
CTO_NUMPROC := $(shell nproc --all)

# docker build extra parameters
DOCKER_BUILD_ARGS=
#DOCKER_BUILD_ARGS="--no-cache"

# Use "yes" below before a multi build to have docker pull the base images using "make build_all" 
DOCKERPULL="no"

# Table below shows driver/CUDA support; for example the 10.2 container needs at least driver 440.33
# https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver
#
# According to https://hub.docker.com/r/nvidia/cuda/
# Looking at the tags, Ubuntu 18.04 is still the primary for 9.x and 10.x
# 
# CUDA 11 came out in May 2020
# Nivida released their CUDA11 containers only with Ubuntu 20.04 support
# https://hub.docker.com/r/nvidia/cuda/tags?page=1&name=20.04
STABLE_CUDA9=9.2
STABLE_CUDA10=10.2
STABLE_CUDA11=11.2.0
# CUDNN needs 5.3 at minimum, extending list from https://en.wikipedia.org/wiki/CUDA#GPUs_supported 
# Skipping Tegra, Jetson, ... (ie not desktop/server GPUs) from this list
# Keeping from Pascal and above
# Also only installing cudnn7 for 18.04 based systems
DNN_ARCH_CUDA9=6.0,6.1
DNN_ARCH_CUDA10=6.0,6.1,7.0,7.5
DNN_ARCH_CUDA11=6.0,6.1,7.0,7.5,8.0,8.6


# According to https://opencv.org/releases/
STABLE_OPENCV3=3.4.13
STABLE_OPENCV4=4.5.1

# TF2 at minimum CUDA 10.1
# TF2 is not going to support CUDA11 until 2.4.0, so not building those yet
##
# According to https://github.com/tensorflow/tensorflow/tags
STABLE_TF1=1.15.5
STABLE_TF2=2.4.1

## Information for build
# https://github.com/bazelbuild/bazelisk
LATEST_BAZELISK=1.7.4
# https://github.com/bazelbuild/bazel
# 20210211 Not using 4.0.0 just yet
LATEST_BAZEL=3.7.2
# https://github.com/keras-team/keras/releases
TF1_KERAS="keras==2.3.1 tensorflow<2"
TF2_KERAS="keras"

# https://github.com/tensorflow/tensorflow/issues/39768
# "use TF 1.15, you have to use Python 3.7 or lower. If you want to use Python 3.8 you have to use TF 2.2 or newer"
TF1_PYTHON=3.7
TF2_PYTHON=3.8

# 20200615: numpy 1.19.0 breaks TF build
# 20201204: numpy >= 1.19.0 still breaks build for TF 1.15.4 & 2.3.1
# 20210211: numpy >= 1.19.0 still breaks build for TF 1.15.5
TF1_NUMPY='numpy<1.19.0'
TF2_NUMPY=numpy

# PyTorch (from pip) using instructions from https://pytorch.org/
PT_CPU="torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html"
PT_CUDA9="torch==1.7.1+cu92 torchvision==0.8.2+cu92 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html"
PT_CUDA10="torch torchvision"


##### CUDA _ Tensorflow _ OpenCV
CTO_BUILDALL =cuda_tensorflow_opencv-${STABLE_CUDA9}_${STABLE_TF1}_${STABLE_OPENCV3}
CTO_BUILDALL+=cuda_tensorflow_opencv-${STABLE_CUDA9}_${STABLE_TF1}_${STABLE_OPENCV4}
CTO_BUILDALL+=cuda_tensorflow_opencv-${STABLE_CUDA9}_${STABLE_TF2}_${STABLE_OPENCV3}
CTO_BUILDALL+=cuda_tensorflow_opencv-${STABLE_CUDA9}_${STABLE_TF2}_${STABLE_OPENCV4}
CTO_BUILDALL+=cuda_tensorflow_opencv-${STABLE_CUDA10}_${STABLE_TF1}_${STABLE_OPENCV3}
CTO_BUILDALL+=cuda_tensorflow_opencv-${STABLE_CUDA10}_${STABLE_TF1}_${STABLE_OPENCV4}
CTO_BUILDALL+=cuda_tensorflow_opencv-${STABLE_CUDA10}_${STABLE_TF2}_${STABLE_OPENCV3}
CTO_BUILDALL+=cuda_tensorflow_opencv-${STABLE_CUDA10}_${STABLE_TF2}_${STABLE_OPENCV4}
CTO_BUILDALL+=cuda_tensorflow_opencv-${STABLE_CUDA11}_${STABLE_TF1}_${STABLE_OPENCV3}
CTO_BUILDALL+=cuda_tensorflow_opencv-${STABLE_CUDA11}_${STABLE_TF1}_${STABLE_OPENCV4}
CTO_BUILDALL+=cuda_tensorflow_opencv-${STABLE_CUDA11}_${STABLE_TF2}_${STABLE_OPENCV3}
CTO_BUILDALL+=cuda_tensorflow_opencv-${STABLE_CUDA11}_${STABLE_TF2}_${STABLE_OPENCV4}

##### CuDNN _ Tensorflow _ OpenCV
DTO_BUILDALL =cudnn_tensorflow_opencv-${STABLE_CUDA9}_${STABLE_TF1}_${STABLE_OPENCV3}
DTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA9}_${STABLE_TF1}_${STABLE_OPENCV4}
# TF > 2.1.0 requires CUDA >= 10.1 -- error when building 2.3.0, skipping CUDNN 9.2
#DTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA9}_${STABLE_TF2}_${STABLE_OPENCV3}
#DTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA9}_${STABLE_TF2}_${STABLE_OPENCV4}
DTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA10}_${STABLE_TF1}_${STABLE_OPENCV3}
DTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA10}_${STABLE_TF1}_${STABLE_OPENCV4}
DTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA10}_${STABLE_TF2}_${STABLE_OPENCV3}
DTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA10}_${STABLE_TF2}_${STABLE_OPENCV4}
# Ubuntu 20.04 comes with Python 3.8, so unable to build TF1 support
#DTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA11}_${STABLE_TF1}_${STABLE_OPENCV3}
#DTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA11}_${STABLE_TF1}_${STABLE_OPENCV4}
DTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA11}_${STABLE_TF2}_${STABLE_OPENCV3}
DTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA11}_${STABLE_TF2}_${STABLE_OPENCV4}

##### Tensorflow _ OpenCV
TO_BUILDALL =tensorflow_opencv-${STABLE_TF1}_${STABLE_OPENCV3}
TO_BUILDALL+=tensorflow_opencv-${STABLE_TF1}_${STABLE_OPENCV4}
TO_BUILDALL+=tensorflow_opencv-${STABLE_TF2}_${STABLE_OPENCV3}
TO_BUILDALL+=tensorflow_opencv-${STABLE_TF2}_${STABLE_OPENCV4}

## By default, provide the list of build targets
all:
	@echo "** Docker Image tag ending: ${CTO_RELEASE}"
	@echo ""
	@echo "** Available Docker images to be built (make targets):"
	@echo "  tensorflow_opencv: "; echo -n "      "; echo ${TO_BUILDALL} | sed -e 's/ /\n      /g'
	@echo "  cuda_tensorflow_opencv: "; echo -n "      "; echo ${CTO_BUILDALL} | sed -e 's/ /\n      /g'
	@echo "  cudnn_tensorflow_opencv: "; echo -n "      "; echo ${DTO_BUILDALL} | sed -e 's/ /\n      /g'
	@echo ""
	@echo "** To build all, use: make build_all"
	@echo ""
	@echo "Note: TensorFlow GPU support can only be compiled for CuDNN containers"

## special command to build all targets
build_all:
	@make ${TO_BUILDALL}
	@make ${CTO_BUILDALL}
	@make ${DTO_BUILDALL}

tensorflow_opencv:
	@make ${TO_BUILDALL}

cuda_tensorflow_opencv:
	@make ${CTO_BUILDALL}

cudnn_tensorflow_opencv:
	@make ${DTO_BUILDALL}

${TO_BUILDALL}:
	@CUDX="" CUDX_COMP="" BTARG="$@" make build_prep

${CTO_BUILDALL}:
	@CUDX="cuda" CUDX_COMP="" BTARG="$@" make build_prep

${DTO_BUILDALL}:
	@CUDX="cudnn" CUDX_COMP="-DWITH_CUDNN=ON -DOPENCV_DNN_CUDA=ON" BTARG="$@" make build_prep
# CUDA_ARCH_BIN and CUDX_FROM are set in build_prep now 

build_prep:
	@$(eval CTO_NAME=$(shell echo ${BTARG} | cut -d- -f 1))
	@$(eval TARGET_VALUE=$(shell echo ${BTARG} | cut -d- -f 2))
	@$(eval CTO_SC=$(shell echo ${TARGET_VALUE} | grep -o "_" | wc -l)) # where 2 means 3 components
	@$(eval CTO_V=$(shell if [ ${CTO_SC} == 1 ]; then echo "0_${TARGET_VALUE}"; else echo "${TARGET_VALUE}"; fi))
	@$(eval CTO_CUDA_VERSION=$(shell echo ${CTO_V} | cut -d_ -f 1))
	@$(eval CTO_CUDA_PRIMEVERSION=$(shell echo ${CTO_CUDA_VERSION} | perl -pe 's/^(\d+\.\d+)\.\d+$$/$$1/;s/\.\d+/.0/'))
	@$(eval CTO_CUDA_USEDVERSION=$(shell echo ${CTO_CUDA_VERSION} | perl -pe 's/^(\d+\.\d+)\.\d+$$/$$1/;s/\./\-/'))
	@$(eval CTO_TENSORFLOW_VERSION=$(shell echo ${CTO_V} | cut -d_ -f 2))
	@$(eval CTO_OPENCV_VERSION=$(shell echo ${CTO_V} | cut -d_ -f 3))

	@$(eval CTO_TMP=${CTO_TENSORFLOW_VERSION})
	@$(eval CTO_TF_CUDNN=$(shell if [ "A${CUDX}" == "Acudnn" ]; then echo "yes"; else echo "no"; fi))
	@$(eval CTO_TF_OPT=$(shell if [ "A${CTO_TMP}" == "A${STABLE_TF1}" ]; then echo "v1"; else echo "v2"; fi))
	@$(eval CTO_TF_KERAS=$(shell if [ "A${CTO_TMP}" == "A${STABLE_TF1}" ]; then echo ${TF1_KERAS}; else echo ${TF2_KERAS}; fi))
	@$(eval CTO_TF_PYTHON=$(shell if [ "A${CTO_TMP}" == "A${STABLE_TF1}" ]; then echo ${TF1_PYTHON}; else echo ${TF2_PYTHON}; fi))
	@$(eval CTO_TF_NUMPY=$(shell if [ "A${CTO_TMP}" == "A${STABLE_TF1}" ]; then echo ${TF1_NUMPY}; else echo ${TF2_NUMPY}; fi))

	@$(eval CTO_TMP=${CTO_TENSORFLOW_VERSION}_${CTO_OPENCV_VERSION}-${CTO_RELEASE})
	@$(eval CTO_TAG=$(shell if [ ${CTO_SC} == 1 ]; then echo ${CTO_TMP}; else echo ${CTO_CUDA_VERSION}_${CTO_TMP}; fi))

	@# Nvidia's container requires Ubuntu 20.04 for CUDA11
	@$(eval CTO_UBUNTU=$(shell if [ "A${CTO_CUDA_VERSION}" == "A${STABLE_CUDA11}" ]; then echo "ubuntu20.04"; else echo "ubuntu18.04"; fi))

	@# 18.04: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/
	@$(eval CTO_TMP18="cuda-npp-${CTO_CUDA_VERSION} cuda-cublas-${CTO_CUDA_PRIMEVERSION} cuda-cufft-${CTO_CUDA_VERSION} cuda-libraries-${CTO_CUDA_VERSION} cuda-npp-dev-${CTO_CUDA_VERSION} cuda-cublas-dev-${CTO_CUDA_PRIMEVERSION} cuda-cufft-dev-${CTO_CUDA_VERSION} cuda-libraries-dev-${CTO_CUDA_VERSION}")
	@$(eval CTO_CUDA_APT=$(shell if [ ${CTO_SC} == 1 ]; then echo ""; else echo ${CTO_TMP}; fi))
	@# 20.04: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/
	@$(eval CTO_TMP20="cuda-libraries-${CTO_CUDA_USEDVERSION} cuda-libraries-dev-${CTO_CUDA_USEDVERSION} cuda-tools-${CTO_CUDA_USEDVERSION} cuda-toolkit-${CTO_CUDA_USEDVERSION} libcublas-${CTO_CUDA_USEDVERSION} libcublas-dev-${CTO_CUDA_USEDVERSION} libcufft-${CTO_CUDA_USEDVERSION} libcufft-dev-${CTO_CUDA_USEDVERSION} libnccl2 libnccl-dev libnpp-${CTO_CUDA_USEDVERSION} libnpp-dev-${CTO_CUDA_USEDVERSION}")
	@# CUDA adds
	@$(eval CTO_CUDA_APT=$(shell if [ ${CTO_SC} == 1 ]; then echo ""; else if [ "A${CTO_UBUNTU}" == "Aubuntu18.04" ]; then echo ${CTO_TMP18}; else echo ${CTO_TMP20}; fi; fi))

	@$(eval CTO_DNN_ARCH=$(shell if [ "A${CTO_CUDA_VERSION}" == "A${STABLE_CUDA9}" ]; then echo "${DNN_ARCH_CUDA9}"; else if [ "A${CTO_CUDA_VERSION}" == "A${STABLE_CUDA10}" ]; then echo "${DNN_ARCH_CUDA10}"; else echo "${DNN_ARCH_CUDA1`}"; fi; fi))
	@$(eval CUDX_COMP=$(shell if [ "A${CUDX}" == "Acudnn" ]; then echo "${CUDX_COMP} -DCUDA_ARCH_BIN=${CTO_DNN_ARCH}"; else echo "${CUDX_COMP}"; fi))

	@$(eval CTO_TMP=$(shell if [ "A${CTO_CUDA_VERSION}" == "A${STABLE_CUDA11}" ]; then echo "-cudnn8"; else echo "-cudnn7"; fi))
	@$(eval CUDX_FROM=$(shell if [ "A${CUDX}" == "Acudnn" ]; then echo "${CTO_TMP}"; else echo ""; fi))

	@$(eval CTO_FROM=$(shell if [ ${CTO_SC} == 1 ]; then echo "ubuntu:18.04"; else echo "nvidia/cuda:${CTO_CUDA_VERSION}${CUDX_FROM}-devel-${CTO_UBUNTU}"; fi))

	@$(eval CTO_TMP="-DWITH_CUDA=ON -DCUDA_FAST_MATH=1 -DWITH_CUBLAS=1 ${CUDX_COMP} -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-${CTO_CUDA_VERSION} -DCUDA_cublas_LIBRARY=cublas -DCUDA_cufft_LIBRARY=cufft -DCUDA_nppim_LIBRARY=nppim -DCUDA_nppidei_LIBRARY=nppidei -DCUDA_nppif_LIBRARY=nppif -DCUDA_nppig_LIBRARY=nppig -DCUDA_nppim_LIBRARY=nppim -DCUDA_nppist_LIBRARY=nppist -DCUDA_nppisu_LIBRARY=nppisu -DCUDA_nppitc_LIBRARY=nppitc -DCUDA_npps_LIBRARY=npps -DCUDA_nppc_LIBRARY=nppc -DCUDA_nppial_LIBRARY=nppial -DCUDA_nppicc_LIBRARY=nppicc -D CUDA_nppicom_LIBRARY=nppicom")
	@$(eval CTO_CUDA_BUILD=$(shell if [ ${CTO_SC} == 1 ]; then echo ""; else echo ${CTO_TMP}; fi))

	@$(eval CTO_PYTORCH=$(shell if [ ${CTO_SC} == 1 ]; then echo "${PT_CPU}"; else if [ "A${CTO_CUDA_VERSION}" == "A${STABLE_CUDA9}" ]; then echo "${PT_CUDA9}"; else echo "${PT_CUDA10}"; fi; fi))

	@echo ""; echo "";
	@echo "[*****] Build: datamachines/${CTO_NAME}:${CTO_TAG}";\

	@if [ "A${DOCKERPULL}" == "Ayes" ]; then echo "** Base image: ${CTO_FROM}"; docker pull ${CTO_FROM}; echo ""; else if [ -f ./${CTO_NAME}-${CTO_TAG}.log ]; then echo "  !! Log file (${CTO_NAME}-${CTO_TAG}.log) exists, skipping rebuild (remove to force)"; echo ""; else CTO_NAME=${CTO_NAME} CTO_TAG=${CTO_TAG} CTO_UBUNTU=${CTO_UBUNTU} CTO_FROM=${CTO_FROM} CTO_TENSORFLOW_VERSION=${CTO_TENSORFLOW_VERSION} CTO_OPENCV_VERSION=${CTO_OPENCV_VERSION} CTO_NUMPROC=$(CTO_NUMPROC) CTO_CUDA_APT="${CTO_CUDA_APT}" CTO_CUDA_BUILD="${CTO_CUDA_BUILD}" CTO_TF_CUDNN="${CTO_TF_CUDNN}" CTO_TF_OPT="${CTO_TF_OPT}" CTO_TF_KERAS="${CTO_TF_KERAS}" CTO_TF_PYTHON="${CTO_TF_PYTHON}" CTO_TF_NUMPY="${CTO_TF_NUMPY}" CTO_DNN_ARCH="${CTO_DNN_ARCH}" CTO_PYTORCH="${CTO_PYTORCH}" make actual_build; fi; fi


actual_build:
	@echo "Press Ctl+c within 5 seconds to cancel"
	@mkdir -p BuildInfo-OpenCV
	@echo "  CTO_FROM               : ${CTO_FROM}" | tee BuildInfo-OpenCV/${CTO_NAME}-${CTO_TAG}.txt
	@for i in 5 4 3 2 1; do echo -n "$$i "; sleep 1; done; echo ""
	# Actual build
	docker build ${DOCKER_BUILD_ARGS} \
	  --build-arg CTO_FROM=${CTO_FROM} \
	  --build-arg CTO_TENSORFLOW_VERSION=${CTO_TENSORFLOW_VERSION} \
	  --build-arg CTO_OPENCV_VERSION=${CTO_OPENCV_VERSION} \
	  --build-arg CTO_NUMPROC=$(CTO_NUMPROC) \
	  --build-arg CTO_CUDA_APT="${CTO_CUDA_APT}" \
	  --build-arg CTO_CUDA_BUILD="${CTO_CUDA_BUILD}" \
	  --build-arg LATEST_BAZELISK="${LATEST_BAZELISK}" \
	  --build-arg LATEST_BAZEL="${LATEST_BAZEL}" \
	  --build-arg CTO_TF_CUDNN="${CTO_TF_CUDNN}" \
	  --build-arg CTO_TF_OPT="${CTO_TF_OPT}" \
	  --build-arg CTO_TF_KERAS="${CTO_TF_KERAS}" \
	  --build-arg CTO_TF_PYTHON="${CTO_TF_PYTHON}" \
	  --build-arg CTO_TF_NUMPY="${CTO_TF_NUMPY}" \
	  --build-arg CTO_DNN_ARCH="${CTO_DNN_ARCH}" \
	  --build-arg CTO_PYTORCH="${CTO_PYTORCH}" \
	  --tag="datamachines/${CTO_NAME}:${CTO_TAG}" \
	  -f ${CTO_UBUNTU}/Dockerfile \
	  . | tee ${CTO_NAME}-${CTO_TAG}.log.temp; exit "$${PIPESTATUS[0]}"
	@mv ${CTO_NAME}-${CTO_TAG}.log.temp ${CTO_NAME}-${CTO_TAG}.log
	@docker run --rm datamachines/${CTO_NAME}:${CTO_TAG} opencv_version -v >> BuildInfo-OpenCV/${CTO_NAME}-${CTO_TAG}.txt
	@mkdir -p BuildInfo-TensorFlow
	@docker run --rm datamachines/${CTO_NAME}:${CTO_TAG} /tmp/tf_info.sh > BuildInfo-TensorFlow/${CTO_NAME}-${CTO_TAG}.txt

clean:
	rm -f *.log.temp

allclean:
	@make clean
	rm -f *.log