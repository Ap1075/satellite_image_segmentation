#!/bin/bash 

# put weights in repo working directory
mkdir ./incubit_data
mkdir ./incubit_data/train ./incubit_data/trainannot
mkdir ./incubit_data/test ./incubit_data/testannot
mkdir ./incubit_data/val ./incubit_data/valannot
mkdir ./incubit_data/json_outs ./incubit_data/weights

cp ./raw/*.png ./incubit_data/train/
cp ./annotation_masks/* ./incubit_data/trainannot/

mv ./incubit_data/train/6_6.png ./incubit_data/train/6_7.png ./incubit_data/train/6_8.png ./incubit_data/train/7_6.png ./incubit_data/train/7_7.png ./incubit_data/train/7_8.png ./incubit_data/train/8_6.png ./incubit_data/train/8_7.png ./incubit_data/train/8_8.png -t ./incubit_data/test

mv ./incubit_data/train/0_6.png ./incubit_data/train/1_7.png ./incubit_data/train/2_8.png ./incubit_data/train/3_6.png ./incubit_data/train/4_7.png -t ./incubit_data/val
mv ./incubit_data/trainannot/0_6.png ./incubit_data/trainannot/1_7.png ./incubit_data/trainannot/2_8.png ./incubit_data/trainannot/3_6.png ./incubit_data/trainannot/4_7.png -t ./incubit_data/valannot
