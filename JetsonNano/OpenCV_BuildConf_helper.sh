#!/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

tag=`make -f Makefile | grep tag | cut -d ':' -f 2`
list=`make -f Makefile| grep -E '\-[[:digit:]]' | tr -s ' ' | cut -d ' ' -f 2`

# Link to the Github release tag (do not forget to tag, push the tag and do the release)
gh="https://github.com/datamachines/cuda_tensorflow_opencv/tree/"
# Base link for OpenCV build dumps
lgb="https://github.com/datamachines/cuda_tensorflow_opencv/blob/master/JetsonNano/OpenCV_BuildConf/"

for i in $list;
do
  v=`echo $i | sed 's/-/:/2'`
  t=`echo $tag | sed 's/\s+//'`

  g="$v $t"

  cont=1
  # Confirm we have a matching file (here for possible future extractions)
  l=`echo $g | perl -pe 's%\:%-%;s%\s%-%'`
  of="${SCRIPTPATH}/OpenCV_BuildConf/$l.txt"; if [ ! -f $of ]; then echo "***** CV: No $of file, skipping"; cont=0; fi

  if [ $cont == 1 ]; then
    tmp=`fgrep FOUND_UBUNTU $of | cut -d " " -f 2`
    ub=`if [ "A$tmp" == "A" ]; then echo "**MISSING**";  else echo $tmp; fi`

    tmp=`fgrep FOUND_CUDNN $of | cut -d " " -f 2`
    dnn=`if [ "A$tmp" == "A" ]; then echo "**MISSING**"; else echo $tmp; fi`

    tmp=`fgrep 'CTO_FROM=' $of | cut -d ":" -f 2 | cut -d "-" -f 1`
    jpr=`if [ "A$tmp" == "A" ]; then echo "**MISSING**"; else echo $tmp; fi`

# j-n_t_o :   | Docker Tag | JetPack | CUDA | CUDNN | TensorFlow | OpenCV | Ubuntu | Github Link | OpenCV Build Conf |
    echo "$g" | perl -ne '@it = ($_ =~ m%^(.+)\:([\d\.]+\_)?([\d\.]+)\_([\d\.]+)\s(\d+)$%); $n = shift @it; $x = pop @it; ($c, $t, $o) = @it; $c=~s%\_$%%; $f= (($c ne "") ? "$c\_": "") . "$t\_$o"; print "| $f-$x | '$jpr' | $c | '$dnn' | $t | $o | '$ub' | [link]('$gh'$x) | [link]('$lgb'$n-$f-$x.txt) |\n";'
  fi
  
done
