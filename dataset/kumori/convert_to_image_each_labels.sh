#!/bin/sh
#dependence : ffmpeg
for label in $(ls -F | grep /);do 
   echo $label
   for file in $(ls $label | grep -E '.*m4v|.*mp4|.*mov|.*MOV');do 
     dir=$(echo $file| sed -E 's/\.mp4|\.m4v|\.mov|\.MOV//g')
     echo $file
     echo $dir
     if [ ! -d $label$dir ]; 
       then mkdir ./$label/$dir
     fi;
     if [ ! -d $label${dir}_cropped ];
        then mkdir ./$label/${dir}_cropped
     fi;
     if [ ! -d $label${dir}_cropped ]; 
       then mkdir ./$label/${dir}_cropped
     fi;
     ffmpeg -i $label/$file -r 5 ./$label/$dir/image_%04d.jpg
   done
done
