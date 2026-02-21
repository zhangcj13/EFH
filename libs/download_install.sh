#! /usr/bin/env bash
EFH_DIR=./EFH #$(pwd)

# use provided tar files
tar -xvf "$EFH_DIR/libs/YOLOX.tar" -C "$EFH_DIR/libs/"
tar -xvf "$EFH_DIR/libs/dagr.tar" -C "$EFH_DIR/libs/"
tar -xvf "$EFH_DIR/libs/dsec-det.tar" -C "$EFH_DIR/libs/"
tar -xvf "$EFH_DIR/libs/detectron2.tar" -C "$EFH_DIR/libs/"

pip install -e $EFH_DIR/libs/YOLOX
pip install -e $EFH_DIR/libs/dagr
pip install -e $EFH_DIR/libs/dsec-det
pip install -e $EFH_DIR/libs/detectron2



