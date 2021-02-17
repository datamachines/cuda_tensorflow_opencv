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
  
  echo "########## Test 1: xeyes will display and disappear after 5 seconds"
  CONTAINER_ID="${CONTAINER_ID}" ../runDocker.sh -N -c timeout -- 5s /usr/bin/xeyes

  echo "########## Test 2: OpenCV to display an image which will disappear after 5 seconds (you can also press any key to continue)"
  CONTAINER_ID="${CONTAINER_ID}" ../runDocker.sh -N -c timeout -- 5s python3 /dmc/cv2_x11.py

done

echo "############################################################"
echo "Done"
