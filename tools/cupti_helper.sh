#!/bin/bash
set -e

BUILD=$1

# /etc/ld.so.conf.d/nvidia.conf point to /usr/local/nvidia which seems to be missing, point to the cuda directory install for libraries
cd /usr/local
if [ -e cuda ]; then
  if [ ! -e nvidia ]; then
    ln -s cuda nvidia
  fi
fi

# CUPTI
tmp="/usr/local/cuda/extras/CUPTI/lib64"

if [[ -d $tmp ]] ; then 
  echo $tmp >> /etc/ld.so.conf.d/nvidia-cupti.conf
  ldconfig
  echo "***** CUPTI added to LD path"
fi

touch /tmp/.${BUILD}_build
