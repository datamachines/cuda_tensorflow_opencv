# Needed SHELL since I'm using zsh
SHELL := /bin/bash
.PHONY: all build_all actual_build build_prep

# Release to match data of Dockerfile and follow YYYYMMDD pattern
CTO_RELEASE=20220521

# Maximize build speed
CTO_NUMPROC := $(shell nproc --all)

# docker build extra parameters
DOCKER_BUILD_ARGS=
#DOCKER_BUILD_ARGS="--no-cache"

# Use "yes" below before a multi build to have docker pull the base images using "make build_all" 
DOCKERPULL="no"

# Use "yes" below to force a TF check post build (recommended)
# this will use docker run [...] --gpus all and extend the TF build log
MLTK_CHECK="yes"

# Table below shows driver/CUDA support; for example the 10.2 container needs at least driver 440.33
# https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver
#
# According to https://hub.docker.com/r/nvidia/cuda/
# Looking at the tags, Ubuntu 18.04 is still the primary for 9.x and 10.x
# 
# CUDA 11 came out in May 2020
# Nivida released their CUDA11 containers only with Ubuntu 20.04 support
# https://hub.docker.com/r/nvidia/cuda/tags?page=1&name=20.04
# 10.2 release now is available with cuddn8
#
# Note: CUDA11 minimum version has to match the one used by PyTorch
#STABLE_CUDA9=9.2
#STABLE_CUDA10=10.2
STABLE_CUDA11p=11.3.1
STABLE_CUDA11l=11.5.2
# For CUDA11 it might be possible to upgrade some of the pre-installed libraries to their latest version, this will add significant space to the container
# to do, uncomment the line below the empty string set
CUDA11_APT_XTRA=""
#CUDA11_APT_XTRA="--allow-change-held-packages"

# CUDNN needs 5.3 at minimum, extending list from https://en.wikipedia.org/wiki/CUDA#GPUs_supported 
# Skipping Tegra, Jetson, ... (ie not desktop/server GPUs) from this list
# Keeping from Pascal and above
# Also only installing cudnn7 for 18.04 based systems
#DNN_ARCH_CUDA9=6.0,6.1,7.0
#DNN_ARCH_CUDA10=6.0,6.1,7.0,7.5
DNN_ARCH_CUDA11=6.0,6.1,7.0,7.5,8.0,8.6

# According to https://opencv.org/releases/
STABLE_OPENCV3=3.4.16
STABLE_OPENCV4=4.5.5

# TF2 at minimum CUDA 10.1
# TF2 CUDA11 minimum is 2.4.0
##
# According to https://github.com/tensorflow/tensorflow/tags
#STABLE_TF1=1.15.5
STABLE_TF2=2.9.0

## Information for build
# https://github.com/bazelbuild/bazelisk
LATEST_BAZELISK=1.11.0
# https://github.com/bazelbuild/bazel (4+ is out but keeping with 3.x for TF < 2.8)
LATEST_BAZEL=5.1.1
# https://github.com/keras-team/keras/releases
# Note: skipping for TF2 < 2.6.0:  built with it
# TF 2.6.0: Keras been split into a separate PIP package (keras), and its code has been moved to the GitHub repositorykeras-team/keras. The API endpoints for tf.keras stay unchanged, but are now backed by the keras PIP package 
#TF1_KERAS="keras==2.3.1 tensorflow<2"
TF2_KERAS="keras"

# https://github.com/tensorflow/tensorflow/issues/39768
# Only for Ubuntu 18.04
#TF1_PYTHON=3.8
TF2_PYTHON=3.8

# 20200615: numpy 1.19.0 breaks TF build
# 20201204: numpy >= 1.19.0 still breaks build for TF 1.15.4 & 2.3.1
# 20210211: numpy >= 1.19.0 breaks TF 1.15.5 + numpy >= 1.20 breaks TF 2.4.1
# 20211027: TF 2.6.0 specifies TF 1.19 so keeping restriction
#TF1_NUMPY='numpy<1.19.0'
#TF2_NUMPY='numpy<1.20.0'
TF2_NUMPY='numpy'

# PyTorch (from pip) using instructions from https://pytorch.org/
# and https://pytorch.org/get-started/previous-versions/
# 1.7.1 last version supported by 9.2
PT_CPU="torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu"
#PT_CUDA9="torch==1.7.1+cu92 torchvision==0.8.2+cu92 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html"
#PT_CUDA10="torch torchvision torchaudio"
PT_CUDA11="torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113"

##########

##### CuDNN _ Tensorflow _ OpenCV (aka CTO)
#CTO_BUILDALL =cudnn_tensorflow_opencv-${STABLE_CUDA9}_${STABLE_TF1}_${STABLE_OPENCV3}
#CTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA9}_${STABLE_TF1}_${STABLE_OPENCV4}
# TF > 2.1.0 requires CUDA >= 10.1 -- error when building 2.3.0, skipping CUDNN 9.2
#CTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA9}_${STABLE_TF2}_${STABLE_OPENCV3}
#CTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA9}_${STABLE_TF2}_${STABLE_OPENCV4}
#CTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA10}_${STABLE_TF1}_${STABLE_OPENCV3}
#CTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA10}_${STABLE_TF1}_${STABLE_OPENCV4}
# Issues with building TF 2.5.0 with CUDA 10.2
# Builld fixed https://github.com/tensorflow/tensorflow/issues/49983 
#CTO_BUILDALL =cudnn_tensorflow_opencv-${STABLE_CUDA10}_${STABLE_TF2}_${STABLE_OPENCV3}
#CTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA10}_${STABLE_TF2}_${STABLE_OPENCV4}
# TF1 does not handle cudnn8 well, skipping
#CTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA11}_${STABLE_TF1}_${STABLE_OPENCV3}
#CTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA11}_${STABLE_TF1}_${STABLE_OPENCV4}
CTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA11p}_${STABLE_TF2}_${STABLE_OPENCV3}
CTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA11p}_${STABLE_TF2}_${STABLE_OPENCV4}
CTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA11l}_${STABLE_TF2}_${STABLE_OPENCV3}
CTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA11l}_${STABLE_TF2}_${STABLE_OPENCV4}

##### Tensorflow _ OpenCV (aka TO)
#TO_BUILDALL =tensorflow_opencv-${STABLE_TF1}_${STABLE_OPENCV3}
#TO_BUILDALL+=tensorflow_opencv-${STABLE_TF1}_${STABLE_OPENCV4}
TO_BUILDALL+=tensorflow_opencv-${STABLE_TF2}_${STABLE_OPENCV3}
TO_BUILDALL+=tensorflow_opencv-${STABLE_TF2}_${STABLE_OPENCV4}

##### Jupyter Notebook ready based on TO & CTO
TO_JUP=jupyter_to-${STABLE_TF2}_${STABLE_OPENCV4}
CTO_JUP=jupyter_cto-${STABLE_CUDA11p}_${STABLE_TF2}_${STABLE_OPENCV4}

## By default, provide the list of build targets
all:
	@echo "**** Docker Image tag ending: ${CTO_RELEASE}"
	@echo ""
	@echo "*** Available Docker images to be built (make targets):"
	@echo "  tensorflow_opencv (aka TO): "; echo -n "      "; echo ${TO_BUILDALL} | sed -e 's/ /\n      /g'
	@echo "  cudnn_tensorflow_opencv (aka CTO): "; echo -n "      "; echo ${CTO_BUILDALL} | sed -e 's/ /\n      /g'
	@echo ""
	@echo "** To build all TO & CTO, use: make build_all"
	@echo ""
	@echo "*** Jupyter Notebook ready containers (requires the base TO & CTO container to be built, will pull otherwise)"
	@echo "  jupyter_to: "; echo -n "      "; echo ${TO_JUP}
	@echo "  jupyter_cto: "; echo -n "      "; echo ${CTO_JUP}
	@echo "  jupyter_all: jupyter_to jupyter_cto"
	@echo ""
	@echo "Note: TensorFlow GPU support can only be compiled for CuDNN containers"

## special command to build all targets
build_all:
	@make ${TO_BUILDALL}
	@make ${CTO_BUILDALL}

tensorflow_opencv:
	@make ${TO_BUILDALL}

cudnn_tensorflow_opencv:
	@make ${CTO_BUILDALL}

${TO_BUILDALL}:
	@CUDX="" CUDX_COMP="" BTARG="$@" make build_prep

${CTO_BUILDALL}:
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

# Two CUDA11 possiblities, work with both
	@$(eval STABLE_CUDA11=$(shell if [ "A${CTO_CUDA_PRIMEVERSION}" == "A11.0" ]; then echo ${CTO_CUDA_VERSION}; else echo "_____"; fi))

# Nvidia's container requires Ubuntu 20.04 for CUDA11 + CPU only now use Ubuntu 20.04, only CUDA9 and CUDA10 uses 18.04
	@$(eval CTO_UBUNTU=$(shell if [ "A${CTO_CUDA_VERSION}" == "A${STABLE_CUDA11}" ]; then echo "ubuntu20.04"; else if [ ${CTO_SC} == 1 ]; then echo "ubuntu20.04"; else echo "ubuntu18.04"; fi; fi))

	@$(eval CTO_TMP=${CTO_TENSORFLOW_VERSION})
	@$(eval CTO_TF_CUDNN=$(shell if [ "A${CUDX}" == "Acudnn" ]; then echo "yes"; else echo "no"; fi))
	@$(eval CTO_TF_OPT=$(shell if [ "A${CTO_TMP}" == "A${STABLE_TF1}" ]; then echo "v1"; else echo "v2"; fi))
	@$(eval CTO_TF_KERAS=$(shell if [ "A${CTO_TMP}" == "A${STABLE_TF1}" ]; then echo ${TF1_KERAS}; else echo ${TF2_KERAS}; fi))

	@$(eval CTO_TF_PYTHON=$(shell if [ "A${CTO_TMP}" == "A${STABLE_TF1}" ]; then echo ${TF1_PYTHON}; else echo ${TF2_PYTHON}; fi))
	@$(eval CTO_TF_PYTHON=$(shell if [ "A${CTO_UBUNTU}" == "Aubuntu18.04" ]; then echo ${CTO_TF_PYTHON}; else echo ""; fi))

	@$(eval CTO_TF_NUMPY=$(shell if [ "A${CTO_TMP}" == "A${STABLE_TF1}" ]; then echo ${TF1_NUMPY}; else echo ${TF2_NUMPY}; fi))

	@$(eval CTO_TMP=${CTO_TENSORFLOW_VERSION}_${CTO_OPENCV_VERSION}-${CTO_RELEASE})
	@$(eval CTO_TAG=$(shell if [ ${CTO_SC} == 1 ]; then echo ${CTO_TMP}; else echo ${CTO_CUDA_VERSION}_${CTO_TMP}; fi))

# 18.04: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/
	@$(eval CTO_TMP18="cuda-npp-${CTO_CUDA_VERSION} cuda-cublas-${CTO_CUDA_PRIMEVERSION} cuda-cufft-${CTO_CUDA_VERSION} cuda-libraries-${CTO_CUDA_VERSION} cuda-npp-dev-${CTO_CUDA_VERSION} cuda-cublas-dev-${CTO_CUDA_PRIMEVERSION} cuda-cufft-dev-${CTO_CUDA_VERSION} cuda-libraries-dev-${CTO_CUDA_VERSION}")
# 20.04: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/
	@$(eval CTO_TMP20=$(shell if [ "A${CUDA11_APT_XTRA}" == "A" ]; then echo ""; else echo "cuda-libraries-${CTO_CUDA_USEDVERSION} cuda-libraries-dev-${CTO_CUDA_USEDVERSION} cuda-tools-${CTO_CUDA_USEDVERSION} cuda-toolkit-${CTO_CUDA_USEDVERSION} libcublas-${CTO_CUDA_USEDVERSION} libcublas-dev-${CTO_CUDA_USEDVERSION} libcufft-${CTO_CUDA_USEDVERSION} libcufft-dev-${CTO_CUDA_USEDVERSION} libnccl2 libnccl-dev libnpp-${CTO_CUDA_USEDVERSION} libnpp-dev-${CTO_CUDA_USEDVERSION}"; fi))
	@$(eval CTO_CUDA_APT=$(shell if [ ${CTO_SC} == 1 ]; then echo ""; else if [ "A${CTO_UBUNTU}" == "Aubuntu18.04" ]; then echo ${CTO_TMP18}; else echo ${CTO_TMP20}; fi; fi))

	@$(eval CTO_DNN_ARCH=$(shell if [ "A${CTO_CUDA_VERSION}" == "A${STABLE_CUDA9}" ]; then echo "${DNN_ARCH_CUDA9}"; else if [ "A${CTO_CUDA_VERSION}" == "A${STABLE_CUDA10}" ]; then echo "${DNN_ARCH_CUDA10}"; else echo "${DNN_ARCH_CUDA11}"; fi; fi))
	@$(eval CTO_DNN_ARCH=$(shell if [ ${CTO_SC} == 1 ]; then echo ""; else echo "${CTO_DNN_ARCH}"; fi))
	@$(eval CUDX_COMP=$(shell if [ ${CTO_SC} == 1 ]; then echo ""; else echo "${CUDX_COMP} -DCUDA_ARCH_BIN=${CTO_DNN_ARCH}"; fi))

# cuddn7 for 9.2 and 10.2 with TF1, cudnn8 otherwise
	@$(eval CTO_TMP=$(shell if [ "A${CTO_CUDA_VERSION}" == "A${STABLE_CUDA9}" ]; then echo "-cudnn7"; else echo "-cudnn8"; fi))
	@$(eval CTO_TMP=$(shell if [ "A${CTO_CUDA_VERSION}" == "A${STABLE_CUDA10}" ] && [ "A${CTO_TENSORFLOW_VERSION}" == "A${STABLE_TF1}" ]; then echo "-cudnn7"; else echo "${CTO_TMP}"; fi))

	@$(eval CUDX_FROM=$(shell if [ "A${CUDX}" == "Acudnn" ]; then echo "${CTO_TMP}"; else echo ""; fi))

	@$(eval CTO_FROM=$(shell if [ ${CTO_SC} == 1 ]; then echo "ubuntu:20.04"; else echo "nvidia/cuda:${CTO_CUDA_VERSION}${CUDX_FROM}-devel-${CTO_UBUNTU}"; fi))

	@$(eval CTO_TMP="-D WITH_CUDA=ON -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda -D CMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs -D CUDA_FAST_MATH=1 -D WITH_CUBLAS=1 ${CUDX_COMP} -D WITH_NVCUVID=ON")
	@$(eval CTO_CUDA_BUILD=$(shell if [ ${CTO_SC} == 1 ]; then echo ""; else echo ${CTO_TMP}; fi))

	@$(eval CTO_PYTORCH=$(shell if [ ${CTO_SC} == 1 ]; then echo "${PT_CPU}"; else if [ "A${CTO_CUDA_VERSION}" == "A${STABLE_CUDA9}" ]; then echo "${PT_CUDA9}"; else if [ "A${CTO_CUDA_VERSION}" == "A${STABLE_CUDA10}" ]; then echo "${PT_CUDA10}"; else echo "${PT_CUDA11}"; fi; fi; fi))

	@echo ""; echo "";
	@echo "[*****] Build: datamachines/${CTO_NAME}:${CTO_TAG}";\

	@if [ "A${DOCKERPULL}" == "Ayes" ]; then echo "** Base image: ${CTO_FROM}"; docker pull ${CTO_FROM}; echo ""; else if [ -f ./${CTO_NAME}-${CTO_TAG}.log ]; then echo "  !! Log file (${CTO_NAME}-${CTO_TAG}.log) exists, skipping rebuild (remove to force)"; echo ""; else CTO_NAME=${CTO_NAME} CTO_TAG=${CTO_TAG} CTO_UBUNTU=${CTO_UBUNTU} CTO_FROM=${CTO_FROM} CTO_TENSORFLOW_VERSION=${CTO_TENSORFLOW_VERSION} CTO_OPENCV_VERSION=${CTO_OPENCV_VERSION} CTO_NUMPROC=$(CTO_NUMPROC) CTO_CUDA_APT="${CTO_CUDA_APT}" CTO_CUDA_BUILD="${CTO_CUDA_BUILD}" CTO_TF_CUDNN="${CTO_TF_CUDNN}" CTO_TF_OPT="${CTO_TF_OPT}" CTO_TF_KERAS="${CTO_TF_KERAS}" CTO_TF_PYTHON="${CTO_TF_PYTHON}" CTO_TF_NUMPY="${CTO_TF_NUMPY}" CTO_CUDA11_APT_XTRA="${CUDA11_APT_XTRA}" CTO_DNN_ARCH="${CTO_DNN_ARCH}" CTO_PYTORCH="${CTO_PYTORCH}" make actual_build; fi; fi


actual_build:
# Build prep
	@mkdir -p BuildInfo-OpenCV
	@mkdir -p BuildInfo-TensorFlow
	@echo ""
	@echo "  CTO_FROM               : ${CTO_FROM}" | tee BuildInfo-OpenCV/${CTO_NAME}-${CTO_TAG}.txt | tee BuildInfo-TensorFlow/${CTO_NAME}-${CTO_TAG}.txt
	@echo ""
	@echo "-- Docker command to be run:"
#	@echo "docker buildx build --progress plain --allow network.host --platform linux/amd64 ${DOCKER_BUILD_ARGS} \\" > ${CTO_NAME}-${CTO_TAG}.cmd
	@echo "docker build ${DOCKER_BUILD_ARGS} \\" > ${CTO_NAME}-${CTO_TAG}.cmd
	@echo "  --build-arg CTO_FROM=\"${CTO_FROM}\" \\" >> ${CTO_NAME}-${CTO_TAG}.cmd
	@echo "  --build-arg CTO_TENSORFLOW_VERSION=\"${CTO_TENSORFLOW_VERSION}\" \\" >> ${CTO_NAME}-${CTO_TAG}.cmd
	@echo "  --build-arg CTO_OPENCV_VERSION=\"${CTO_OPENCV_VERSION}\" \\" >> ${CTO_NAME}-${CTO_TAG}.cmd
	@echo "  --build-arg CTO_NUMPROC=\"$(CTO_NUMPROC)\" \\" >> ${CTO_NAME}-${CTO_TAG}.cmd
	@echo "  --build-arg CTO_CUDA_APT=\"${CTO_CUDA_APT}\" \\" >> ${CTO_NAME}-${CTO_TAG}.cmd
	@echo "  --build-arg CTO_CUDA_BUILD=\"${CTO_CUDA_BUILD}\" \\" >> ${CTO_NAME}-${CTO_TAG}.cmd
	@echo "  --build-arg LATEST_BAZELISK=\"${LATEST_BAZELISK}\" \\" >> ${CTO_NAME}-${CTO_TAG}.cmd
	@echo "  --build-arg LATEST_BAZEL=\"${LATEST_BAZEL}\" \\" >> ${CTO_NAME}-${CTO_TAG}.cmd
	@echo "  --build-arg CTO_TF_CUDNN=\"${CTO_TF_CUDNN}\" \\" >> ${CTO_NAME}-${CTO_TAG}.cmd
	@echo "  --build-arg CTO_TF_OPT=\"${CTO_TF_OPT}\" \\" >> ${CTO_NAME}-${CTO_TAG}.cmd
	@echo "  --build-arg CTO_TF_KERAS=\"${CTO_TF_KERAS}\" \\" >> ${CTO_NAME}-${CTO_TAG}.cmd
	@echo "  --build-arg CTO_TF_PYTHON=\"${CTO_TF_PYTHON}\" \\" >> ${CTO_NAME}-${CTO_TAG}.cmd
	@echo "  --build-arg CTO_TF_NUMPY=\"${CTO_TF_NUMPY}\" \\" >> ${CTO_NAME}-${CTO_TAG}.cmd
	@echo "  --build-arg CTO_DNN_ARCH=\"${CTO_DNN_ARCH}\" \\" >> ${CTO_NAME}-${CTO_TAG}.cmd
	@echo "  --build-arg CTO_CUDA11_APT_XTRA=\"${CTO_CUDA11_APT_XTRA}\" \\" >> ${CTO_NAME}-${CTO_TAG}.cmd
	@echo "  --build-arg CTO_PYTORCH=\"${CTO_PYTORCH}\" \\" >> ${CTO_NAME}-${CTO_TAG}.cmd
	@echo "  --tag=\"datamachines/${CTO_NAME}:${CTO_TAG}\" \\" >> ${CTO_NAME}-${CTO_TAG}.cmd
	@echo "  -f ${CTO_UBUNTU}/Dockerfile \\" >> ${CTO_NAME}-${CTO_TAG}.cmd
	@echo "  ." >> ${CTO_NAME}-${CTO_TAG}.cmd
	@cat ${CTO_NAME}-${CTO_TAG}.cmd | tee ${CTO_NAME}-${CTO_TAG}.log.temp | tee -a BuildInfo-OpenCV/${CTO_NAME}-${CTO_TAG}.txt | tee -a BuildInfo-TensorFlow/${CTO_NAME}-${CTO_TAG}.txt
	@echo "" | tee -a ${CTO_NAME}-${CTO_TAG}.log.temp
	@echo "Press Ctl+c within 5 seconds to cancel"
	@for i in 5 4 3 2 1; do echo -n "$$i "; sleep 1; done; echo ""
# Actual build
	@chmod +x ./${CTO_NAME}-${CTO_TAG}.cmd
	@./${CTO_NAME}-${CTO_TAG}.cmd | tee -a ${CTO_NAME}-${CTO_TAG}.log.temp; exit "$${PIPESTATUS[0]}"
	@fgrep "CUDA NVCC" ${CTO_NAME}-${CTO_TAG}.log.temp >> BuildInfo-OpenCV/${CTO_NAME}-${CTO_TAG}.txt || true
	@docker run --rm datamachines/${CTO_NAME}:${CTO_TAG} opencv_version -v >> BuildInfo-OpenCV/${CTO_NAME}-${CTO_TAG}.txt
	@docker run --rm datamachines/${CTO_NAME}:${CTO_TAG} /tmp/tf_info.sh >> BuildInfo-TensorFlow/${CTO_NAME}-${CTO_TAG}.txt
	@if [ "A${MLTK_CHECK}" == "Ayes" ]; then CTO_NAME=${CTO_NAME} CTO_TAG=${CTO_TAG} make force_mltk_check; fi
	@mv ${CTO_NAME}-${CTO_TAG}.log.temp ${CTO_NAME}-${CTO_TAG}.log
	@rm -f ./${CTO_NAME}-${CTO_TAG}.cmd

##### Force ML Toolkit checks
force_mltk_check:
	@docker run --rm -v `pwd`:/dmc --gpus all datamachines/${CTO_NAME}:${CTO_TAG} python3 /dmc/test/tf_hw.py | tee -a BuildInfo-TensorFlow/${CTO_NAME}-${CTO_TAG}.txt; exit "$${PIPESTATUS[0]}"

##### Jupyter Notebook
JN_MODE=""
JN_UID=$(shell id -u)
JN_GID=$(shell id -g)

jupyter_all:
	@make jupyter_to
	@make jupyter_cto

jupyter_to:
	@make ${TO_JUP}

jupyter_cto:
	@make ${CTO_JUP}

${TO_JUP}:
	@$(eval JN=$(shell echo ${TO_JUP} | sed 's/-/:/'))
	@$(eval JB=$(shell echo ${JN} | cut -d : -f 1))
	@$(eval JT=$(shell echo ${JN} | cut -d : -f 2))
	@$(eval JN="${JB}${JN_MODE}:${JT}")
	@cd Jupyter_build; docker build --build-arg JUPBC="datamachines/tensorflow_opencv:${JT}-${CTO_RELEASE}" --build-arg JUID=${JN_UID} --build-arg JGID=${JN_GID} -f Dockerfile${JN_MODE} --tag="datamachines/${JN}-${CTO_RELEASE}" .

${CTO_JUP}:
	@$(eval JN=$(shell echo ${CTO_JUP} | sed 's/-/:/'))
	@$(eval JB=$(shell echo ${JN} | cut -d : -f 1))
	@$(eval JT=$(shell echo ${JN} | cut -d : -f 2))
	@$(eval JN="${JB}${JN_MODE}:${JT}")
	@cd Jupyter_build; docker build --build-arg JUPBC="datamachines/cudnn_tensorflow_opencv:${JT}-${CTO_RELEASE}" --build-arg JUID=${JN_UID} --build-arg JGID=${JN_GID} -f Dockerfile${JN_MODE} --tag="datamachines/${JN}-${CTO_RELEASE}" .

##### Various cleanup
clean:
	rm -f *.log.temp

allclean:
	@make clean
	rm -f *.log