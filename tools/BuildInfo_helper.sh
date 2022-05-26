#!/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

tag=`make -f ${SCRIPTPATH}/../Makefile | grep tag | cut -d ':' -f 2`
list=`make -f ${SCRIPTPATH}/../Makefile| grep -v jupyter | grep '-' | tr -s ' ' | cut -d ' ' -f 2`

# Link to the Github released tag (do not forget to tag, push the tag and do the release)
gh="https://github.com/datamachines/cuda_tensorflow_opencv/tree/"
# Base link for OpenCV build info
lgb="https://github.com/datamachines/cuda_tensorflow_opencv/blob/master/BuildInfo-OpenCV/"
# Base link for TensorFlow build info
tgb="https://github.com/datamachines/cuda_tensorflow_opencv/blob/master/BuildInfo-TensorFlow/"


skipped=0
missing=0
for i in $list;
do
  v=`echo $i | sed 's/-/:/'`
  t=`echo $tag | sed 's/\s+//'`

  g="$v $t"

  cont=1
  # Confirm we have a matching file (here for possible future extractions)
  l=`echo $g | perl -pe 's%\:%-%;s%\s%-%'`
  of="${SCRIPTPATH}/../BuildInfo-OpenCV/$l.txt"; if [ ! -f $of ]; then echo "***** CV: No $of file, skipping"; cont=0; fi
  tf="${SCRIPTPATH}/../BuildInfo-TensorFlow/$l.txt"; if [ ! -f $tf ]; then echo "***** TF: No $tf file, skipping"; cont=0; fi

  if [ $cont == 1 ]; then
    tmp=`fgrep FOUND_UBUNTU $tf | cut -d " " -f 2`
    ub=`if [ "A$tmp" == "A" ]; then echo "**MISSING**";  else echo $tmp; fi`

    tmp=`fgrep FOUND_CUDNN $tf | cut -d " " -f 2`
    dnn=`if [ "A$tmp" == "A" ]; then echo "**MISSING**"; else echo $tmp; fi`

    # t_o:      | Docker Tag | TensorFlow | OpenCV | Ubuntu | Github Link | OpenCV Build Info | TensorFlow Build Info |
    # c_t_o :   | Docker Tag | CUDA | TensorFlow | OpenCV | Ubuntu | Github Link | OpenCV Build Info | TensorFlow Build Info |
    # n_t_o :   | Docker Tag | CUDA | CUDNN | TensorFlow | OpenCV | Ubuntu | Github Link | OpenCV Build Info | TensorFlow Build Info |
    #echo "[$v-$t]"
    line=`echo "$g" | perl -ne '@it = ($_ =~ m%^(.+)\:([\d\.]+\_)?([\d\.]+)\_([\d\.]+)\s(\d+)$%); $n = shift @it; $x = pop @it; ($c, $t, $o) = @it; $c=~s%\_$%%; $f= (($c ne "") ? "$c\_": "") . "$t\_$o"; print "| $f-$x "; print "| $c " if ($c != 0); print "| '$dnn' " if ($n =~ m%^cudnn_%); print "| $t | $o | '$ub' | [link]('$gh'$x) | [link]('$lgb'$n-$f-$x.txt) | [link]('$tgb'$n-$f-$x.txt) |\n";'`
    echo $line
    check=`echo $line | grep '**MISSING**' | wc -l`
    if [ $check == 1 ]; then
      missing=$((missing+1))
    fi
  else
    skipped=$((skipped+1))
  fi
done

if [ $skipped -gt 0 ]; then
  echo ""
  echo "!!!!! Warning: Skipped $skipped potential entries"
fi
if [ $missing -gt 0 ]; then
  echo ""
  echo "!!!!! Warning: $missing entry is missing a valid CV and/or TF value"
fi
