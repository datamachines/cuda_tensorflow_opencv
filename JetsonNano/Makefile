# Needed SHELL since I'm using zsh
SHELL := /bin/bash
.PHONY: all build_all actual_build build_prep

# Release to match data of Dockerfile and follow YYYYMMDD pattern
CTO_RELEASE=20210218

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

##########
JETPACK_RELEASE="32.5.0"
STABLE_CUDA10=10.2
ARCH_CUDA10=5.3

# According to https://opencv.org/releases/
STABLE_OPENCV3=3.4.13
STABLE_OPENCV4=4.5.1

# Following releases in https://ngc.nvidia.com/catalog/containers/nvidia:l4t-tensorflow
# Not compiling TF on the nano, simply use the "ready" container
STABLE_TF1=1.15
STABLE_TF2=2.3

# https://github.com/keras-team/keras/releases
TF1_KERAS="keras==2.3.1 tensorflow<2"
TF2_KERAS="keras"

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

build_prep:
	@$(eval CTO_NAME=$(shell echo ${BTARG} | cut -d- -f 1-2))
	@$(eval TARGET_VALUE=$(shell echo ${BTARG} | cut -d- -f 3))
	@$(eval CTO_SC=$(shell echo ${TARGET_VALUE} | grep -o "_" | wc -l)) # where 2 means 3 components
	@$(eval CTO_V=$(shell if [ ${CTO_SC} == 1 ]; then echo "0_${TARGET_VALUE}"; else echo "${TARGET_VALUE}"; fi))
	@$(eval CTO_CUDA_VERSION=$(shell echo ${CTO_V} | cut -d_ -f 1))
	@$(eval CTO_CUDA_PRIMEVERSION=$(shell echo ${CTO_CUDA_VERSION} | perl -pe 's/\.\d+/.0/'))
	@$(eval CTO_TENSORFLOW_VERSION=$(shell echo ${CTO_V} | cut -d_ -f 2))
	@$(eval CTO_OPENCV_VERSION=$(shell echo ${CTO_V} | cut -d_ -f 3))

	@$(eval CTO_FROM="nvcr.io/nvidia/l4t-tensorflow:r${JETPACK_RELEASE}-tf${CTO_TENSORFLOW_VERSION}-py3")

	@$(eval CTO_TAG=$(shell echo "${CTO_CUDA_VERSION}_${CTO_TENSORFLOW_VERSION}_${CTO_OPENCV_VERSION}-${CTO_RELEASE}"))

	@$(eval CTO_TF_KERAS=$(shell if [ "A${CTO_TENSORFLOW_VERSION}" == "A${STABLE_TF1}" ]; then echo ${TF1_KERAS}; else echo ${TF2_KERAS}; fi))

	@$(eval DNN_ARCH=$(shell echo "${ARCH_CUDA10}"))
	@$(eval CUDX_COMP=$(shell echo "${CUDX_COMP} -DCUDA_ARCH_BIN=${DNN_ARCH}"))

	@$(eval CTO_CUDA_BUILD=$(shell echo "-DWITH_CUDA=ON -DCUDA_FAST_MATH=1 -DWITH_CUBLAS=1 ${CUDX_COMP} -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-${CTO_CUDA_VERSION} -DCUDA_ARCH_PTX=\"\""))

	@echo ""; echo ""
	@echo "[*****] About to build datamachines/${CTO_NAME}:${CTO_TAG}"

	@if [ "A${DOCKERPULL}" == "Ayes" ]; then echo "** Base image: ${CTO_FROM}"; docker pull ${CTO_FROM}; echo ""; else if [ -f ./${CTO_NAME}-${CTO_TAG}.log ]; then echo "  !! Log file (${CTO_NAME}-${CTO_TAG}.log) exists, skipping rebuild (remove to force)"; echo ""; else CTO_NAME=${CTO_NAME} CTO_TAG=${CTO_TAG} CTO_FROM=${CTO_FROM} CTO_OPENCV_VERSION=${CTO_OPENCV_VERSION} CTO_NUMPROC=$(CTO_NUMPROC) CTO_CUDA_BUILD="${CTO_CUDA_BUILD}" CTO_TF_KERAS="${CTO_TF_KERAS}" make actual_build; fi; fi


actual_build:
	@mkdir -p OpenCV_BuildConf
	@echo ""
	@echo "  CTO_FROM               : ${CTO_FROM}" | tee OpenCV_BuildConf/${CTO_NAME}-${CTO_TAG}.txt
	@echo ""
	@echo "-- Docker command to be run:"
	@echo "docker build ${DOCKER_BUILD_ARGS} \\" > ${CTO_NAME}-${CTO_TAG}.cmd
	@echo "  --build-arg CTO_FROM=\"${CTO_FROM}\" \\" >> ${CTO_NAME}-${CTO_TAG}.cmd
	@echo "  --build-arg CTO_OPENCV_VERSION=${CTO_OPENCV_VERSION} \\" >> ${CTO_NAME}-${CTO_TAG}.cmd
	@echo "  --build-arg CTO_NUMPROC=$(CTO_NUMPROC) \\" >> ${CTO_NAME}-${CTO_TAG}.cmd
	@echo "  --build-arg CTO_CUDA_BUILD=\"${CTO_CUDA_BUILD}\" \\" >> ${CTO_NAME}-${CTO_TAG}.cmd
	@echo "  --build-arg CTO_TF_KERAS=\"${CTO_TF_KERAS}\" \\" >> ${CTO_NAME}-${CTO_TAG}.cmd
	@echo "  --tag=\"datamachines/${CTO_NAME}:${CTO_TAG}\" \\" >> ${CTO_NAME}-${CTO_TAG}.cmd
	@echo "  ." >> ${CTO_NAME}-${CTO_TAG}.cmd
	@cat ${CTO_NAME}-${CTO_TAG}.cmd | tee ${CTO_NAME}-${CTO_TAG}.log.temp | tee -a OpenCV_BuildConf/${CTO_NAME}-${CTO_TAG}.txt
	@echo "" | tee -a ${CTO_NAME}-${CTO_TAG}.log.temp
	@echo "Press Ctl+c within 5 seconds to cancel"
	@for i in 5 4 3 2 1; do echo -n "$$i "; sleep 1; done; echo ""
# Actual build
	@chmod +x ./${CTO_NAME}-${CTO_TAG}.cmd
	@./${CTO_NAME}-${CTO_TAG}.cmd | tee ${CTO_NAME}-${CTO_TAG}.log.temp; exit "$${PIPESTATUS[0]}"
	@docker run --rm datamachines/${CTO_NAME}:${CTO_TAG} opencv_version -v >> OpenCV_BuildConf/${CTO_NAME}-${CTO_TAG}.txt
	@docker run --rm -it -v `pwd`/..:/dmc --gpus all --runtime nvidia datamachines/${CTO_NAME}:${CTO_TAG} /dmc/tools/tf_info.sh >> OpenCV_BuildConf/${CTO_NAME}-${CTO_TAG}.txt
	@if [ "A${MLTK_CHECK}" == "Ayes" ]; then CTO_NAME=${CTO_NAME} CTO_TAG=${CTO_TAG} make force_mltk_check; fi
	@mv ${CTO_NAME}-${CTO_TAG}.log.temp ${CTO_NAME}-${CTO_TAG}.log
	@rm -f ./${CTO_NAME}-${CTO_TAG}.cmd

##### Force ML Toolkit checks
force_mltk_check:
	@docker run --rm -it -v `pwd`/..:/dmc --gpus all --runtime nvidia datamachines/${CTO_NAME}:${CTO_TAG} python3 /dmc/test/tf_hw.py >> OpenCV_BuildConf/${CTO_NAME}-${CTO_TAG}.txt


##### Various cleanup
clean:
	@rm -f *.log.temp

allclean:
	@make clean
	@rm -f *.log