#!/bin/bash

# Expected to be run in container
if [ ! -f /.within_container ]; then
  echo "Tool expected to be ran within the container, aborting"
  exit 1
fi

echo "[pip list]"
pip3 list
echo ""
echo -n "-- Confirming OpenCV Python is installed. Version: "
python3 -c 'import cv2; print(cv2.__version__)'

if [ -f /tmp/tf_env.dump ]; then
  echo ""
  echo "-------------------------------------------------------"
  echo ""

  echo "[TensorFlow build information]"
  cat /tmp/tf_env.dump
fi

echo ""
echo "-------------------------------------------------------"
echo ""

echo "[Extra information]"

# Ubuntu version
tmp="/etc/lsb-release"
if [ ! -f $tmp ]; then
  echo "Unable to confirm Ubuntu version, aborting"
  exit 1
fi

echo -n "FOUND_UBUNTU: "
perl -ne 'print $1 if (m%DISTRIB_RELEASE=(.+)%)' $tmp
echo ""

# CUDNN version
cudnn_inc="/usr/include/cudnn.h"
cudnn8_inc="/usr/include/x86_64-linux-gnu/cudnn_version_v8.h"
if [ -f $cudnn8_inc ]; then
  cudnn_inc="${cudnn8_inc}"
fi
if [ ! -f $cudnn_inc ]; then
  cudnn="Not_Available"
else
  cmj="$(sed -n 's/^#define CUDNN_MAJOR\s*\(.*\).*/\1/p' $cudnn_inc)"
  cmn="$(sed -n 's/^#define CUDNN_MINOR\s*\(.*\).*/\1/p' $cudnn_inc)"
  cpl="$(sed -n 's/^#define CUDNN_PATCHLEVEL\s*\(.*\).*/\1/p' $cudnn_inc)"
  cudnn="${cmj}.${cmn}.${cpl}"
fi
echo "FOUND_CUDNN: $cudnn"
