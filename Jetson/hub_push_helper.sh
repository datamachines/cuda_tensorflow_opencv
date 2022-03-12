#!/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

tag=`make -f ${SCRIPTPATH}/Makefile | grep tag | cut -d ':' -f 2`
list=`make -f ${SCRIPTPATH}/Makefile| grep '-' | tr -s ' ' | cut -d ' ' -f 2`

# Make sure to be `docker login`-ed

echo "***** Getting list from Makefile"
todo=""
for i in $list;
do
  v=`echo $i | sed 's/-/:/'`
  t=`echo $tag | sed 's/\s+//'`

  cid="datamachines/$v-$t"
  echo " - $cid"
  todo="$todo $cid"  
done

echo ""
echo "Press Ctl+c within 5 seconds to cancel"
for i in 5 4 3 2 1; do echo -n "$i "; sleep 1; done; echo ""

for cid in $todo;
do
  echo ""
  echo ""
  echo "***** Pushing: $cid"
  echo ""
  docker push $cid || exit 1
done

echo "Done"