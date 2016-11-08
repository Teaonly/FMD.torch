#!/bin/bash

rm all.list
touch all.list

## voc database
find ./VOCdevkit/VOC2007/JPEGImages/ -name *.jpg | sort > /tmp/i.txt
find ./VOCdevkit/VOC2007/Annotations/ -name *.xml | sort > /tmp/a.txt
paste /tmp/i.txt /tmp/a.txt > /tmp/all.txt
sed -i -e 's/$/\ voc/' /tmp/all.txt
cat /tmp/all.txt >> all.list

find ./VOCdevkit/VOC2012/JPEGImages/ -name *.jpg | sort > /tmp/i.txt
find ./VOCdevkit/VOC2012/Annotations/ -name *.xml | sort > /tmp/a.txt
paste /tmp/i.txt /tmp/a.txt > /tmp/all.txt
sed -i -e 's/$/\ voc/' /tmp/all.txt
cat /tmp/all.txt >> all.list

