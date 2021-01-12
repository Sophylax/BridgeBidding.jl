#!/bin/bash
data="${1}"
prefix="./"
file="${prefix}${2}"

mkfifo $file
echo -e $data > $file &
./bridge -d $file
rm $file