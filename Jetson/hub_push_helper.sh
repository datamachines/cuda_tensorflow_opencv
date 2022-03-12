#!/bin/bash

tag=`make -f Makefile | grep tag | cut -d ':' -f 2`
list=`make -f Makefile| grep -E '\-[[:digit:]]' | tr -s ' ' | cut -d ' ' -f 2`

# Make sure to be `docker login`-ed

for i in $list;
do
  v=`echo $i | sed 's/-/:/2'`
  t=`echo $tag | sed 's/\s+//'`

  cid="datamachines/$v-$t"

  echo ""
  echo "##### $cid"
  docker push $cid || exit 1
  echo ""

done

echo "Done"
