#!/bin/bash

# Make the directory.
BASE_DIR="/opt/cocoapi"
if [ ! -d "${BASE_DIR}" ] ; then
	mkdir -p "${BASE_DIR}"
fi
cd "${BASE_DIR}" || exit

# Prepare for annotations.
curl -O http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip -q annotations_trainval2014.zip
http://images.cocodataset.org/annotations/image_info_test2014.zip
unzip -q image_info_test2014.zip

# Prepare for images.
if [ ! -d "${BASE_DIR}/images" ] ; then
	mkdir -p "${BASE_DIR}/images"
fi
cd "${BASE_DIR}/images" || exit
curl -O http://images.cocodataset.org/zips/train2014.zip
curl -O http://images.cocodataset.org/zips/val2014.zip
curl -O http://images.cocodataset.org/zips/test2014.zip
unzip -q train2014.zip
unzip -q val2014.zip
unzip -q test2014.zip
