#!/bin/bash

tag=`make -f ../Makefile | grep tag | cut -d ':' -f 2`
list=`make -f ../Makefile| grep '-' | tr -s ' ' | cut -d ' ' -f 2`

for i in $list;
do
  v=`echo $i | sed 's/-/:/'`
  t=`echo $tag | sed 's/\s+//'`
  CONTAINER_ID="datamachines/$v-$t"

  echo ""
  echo "############################################################"
  echo "#################### X11 test for ${CONTAINER_ID}"
  
  echo "########## Test 1: xeyes will display, close the window to continue"
  CONTAINER_ID="${CONTAINER_ID}" ../runDocker.sh -N -c /usr/bin/xeyes

  echo "########## Test 2: OpenCV to display an image, close the window (press q) to continue"
  CONTAINER_ID="${CONTAINER_ID}" ../runDocker.sh -N -c python3 -- /dmc/cv2_x11.py

done

echo "############################################################"
echo "Done"
