# Needed SHELL since I'm using zsh
SHELL := /bin/bash
.PHONY: all build_all actual_build build_prep

# Release to match data of Dockerfile and follow YYYYMMDD pattern
CTO_RELEASE=20200423

# Maximize build speed
CTO_NUMPROC := $(shell nproc --all)

# Jetpack 4.3 <-> L4T 32.3.1
# provides:
# CUDA 10.0
# ARCH = 5.3
CTO_FROM="nvcr.io/nvidia/l4t-base:r32.3.1"
CTO_CUDA_APT=""
STABLE_CUDA10=10.0
ARCH_CUDA10=5.3

# According to https://opencv.org/releases/
STABLE_OPENCV3=3.4.10
STABLE_OPENCV4=4.3.0

# Following releases in https://developer.download.nvidia.com/compute/redist/jp/v43/tensorflow/
# Note that we are using the `tensorflow` entry because `tensorflow-gpu` is now part of it
STABLE_TF1=1.15.2
STABLE_TF2=2.1.0
NV_VERSION=nv20.3

##### JetsonNano - CUDA _ Tensorflow _ OpenCV
CTO_BUILDALL =jetsonnano-cuda_tensorflow_opencv-${STABLE_CUDA10}_${STABLE_TF1}_${STABLE_OPENCV3}
CTO_BUILDALL+=jetsonnano-cuda_tensorflow_opencv-${STABLE_CUDA10}_${STABLE_TF1}_${STABLE_OPENCV4}
CTO_BUILDALL+=jetsonnano-cuda_tensorflow_opencv-${STABLE_CUDA10}_${STABLE_TF2}_${STABLE_OPENCV3}
CTO_BUILDALL+=jetsonnano-cuda_tensorflow_opencv-${STABLE_CUDA10}_${STABLE_TF2}_${STABLE_OPENCV4}

## By default, provide the list of build targets
all:
	@echo "** Docker Image tag ending: ${CTO_RELEASE}"
	@echo ""
	@echo "** Available Docker images to be built (make targets):"
	@echo "  jetsonnano-cuda_tensorflow_opencv: "; echo -n "      "; echo ${CTO_BUILDALL} | sed -e 's/ /\n      /g'
	@echo ""
	@echo "** To build all, use: make build_all"

## special command to build all targets
build_all:
	@make ${CTO_BUILDALL}

jetsonnano-cuda_tensorflow_opencv:
	@make ${CTO_BUILDALL}

${CTO_BUILDALL}:
	@CUDX="cuda" CUDX_COMP="" BTARG="$@" make build_prep
# CUDA_ARCH_BIN is set in build_prep now 

build_prep:
	@$(eval CTO_NAME=$(shell echo ${BTARG} | cut -d- -f 1-2))
	@$(eval TARGET_VALUE=$(shell echo ${BTARG} | cut -d- -f 3))
	@$(eval CTO_SC=$(shell echo ${TARGET_VALUE} | grep -o "_" | wc -l)) # where 2 means 3 components
	@$(eval CTO_V=$(shell if [ ${CTO_SC} == 1 ]; then echo "0_${TARGET_VALUE}"; else echo "${TARGET_VALUE}"; fi))
	@$(eval CTO_CUDA_VERSION=$(shell echo ${CTO_V} | cut -d_ -f 1))
	@$(eval CTO_CUDA_PRIMEVERSION=$(shell echo ${CTO_CUDA_VERSION} | perl -pe 's/\.\d+/.0/'))
	@$(eval CTO_TENSORFLOW_VERSION=$(shell echo ${CTO_V} | cut -d_ -f 2))
	@$(eval CTO_OPENCV_VERSION=$(shell echo ${CTO_V} | cut -d_ -f 3))

	@$(eval CTO_TMP=${CTO_TENSORFLOW_VERSION})
	@$(eval CTO_TENSORFLOW_PYTHON=$(shell if [ "A${CTO_TMP}" == "A${STABLE_TF1}" ]; then echo "tensorflow==${STABLE_TF1}"; else echo "tensorflow==${STABLE_TF2}"; fi))
	@$(eval CTO_TENSORFLOW_PYTHON="${CTO_TENSORFLOW_PYTHON}+${NV_VERSION}")

	@$(eval CTO_TMP=${CTO_TENSORFLOW_VERSION}_${CTO_OPENCV_VERSION}-${CTO_RELEASE})
	@$(eval CTO_TAG=$(shell if [ ${CTO_SC} == 1 ]; then echo ${CTO_TMP}; else echo ${CTO_CUDA_VERSION}_${CTO_TMP}; fi))

	@$(eval DNN_ARCH=$(shell echo "${ARCH_CUDA10}"))
	@$(eval CUDX_COMP=$(shell echo "${CUDX_COMP} -DCUDA_ARCH_BIN=${DNN_ARCH}"))

	@$(eval CTO_TMP="-DWITH_CUDA=ON -DCUDA_FAST_MATH=1 -DWITH_CUBLAS=1 ${CUDX_COMP} -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-${CTO_CUDA_VERSION} -DCUDA_ARCH_PTX=\"\"")
	@$(eval CTO_CUDA_BUILD=$(shell if [ ${CTO_SC} == 1 ]; then echo ""; else echo ${CTO_TMP}; fi))

	@echo ""; echo ""
	@echo "[*****] About to build datamachines/${CTO_NAME}:${CTO_TAG}"

	@if [ -f ./${CTO_NAME}-${CTO_TAG}.log ]; then echo "  !! Log file (${CTO_NAME}-${CTO_TAG}.log) exists, skipping rebuild (remove to force)"; echo ""; else CTO_NAME=${CTO_NAME} CTO_TAG=${CTO_TAG} CTO_FROM=${CTO_FROM} CTO_TENSORFLOW_PYTHON=${CTO_TENSORFLOW_PYTHON} CTO_OPENCV_VERSION=${CTO_OPENCV_VERSION} CTO_NUMPROC=$(CTO_NUMPROC) CTO_CUDA_APT="${CTO_CUDA_APT}" CTO_CUDA_BUILD="${CTO_CUDA_BUILD}" make actual_build; fi


actual_build:
	@echo "Press Ctl+c within 5 seconds to cancel"
	@echo "  CTO_FROM               : ${CTO_FROM}" | tee OpenCV_BuildConf/${CTO_NAME}-${CTO_TAG}.txt
	@echo "  CTO_TENSORFLOW_PYTHON  : ${CTO_TENSORFLOW_PYTHON}" | tee -a OpenCV_BuildConf/${CTO_NAME}-${CTO_TAG}.txt
	@for i in 5 4 3 2 1; do echo -n "$$i "; sleep 1; done; echo ""
	docker build \
	  --build-arg CTO_FROM=${CTO_FROM} \
	  --build-arg CTO_TENSORFLOW_PYTHON=${CTO_TENSORFLOW_PYTHON} \
	  --build-arg CTO_OPENCV_VERSION=${CTO_OPENCV_VERSION} \
	  --build-arg CTO_NUMPROC=$(CTO_NUMPROC) \
	  --build-arg CTO_CUDA_APT="${CTO_CUDA_APT}" \
	  --build-arg CTO_CUDA_BUILD="${CTO_CUDA_BUILD}" \
	  --tag="datamachines/${CTO_NAME}:${CTO_TAG}" \
	  . | tee ${CTO_NAME}-${CTO_TAG}.log.temp; exit "$${PIPESTATUS[0]}"
	@mv ${CTO_NAME}-${CTO_TAG}.log.temp ${CTO_NAME}-${CTO_TAG}.log
	@mkdir -p OpenCV_BuildConf
	@docker run --rm datamachines/${CTO_NAME}:${CTO_TAG} opencv_version -v >> OpenCV_BuildConf/${CTO_NAME}-${CTO_TAG}.txt
	