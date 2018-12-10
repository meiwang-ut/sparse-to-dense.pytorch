#!/usr/bin/env bash

## small dataset on eldar-1
#python3 main.py -a resnet18 -d deconv2 -m rgb -s 0
#python3 main.py -a resnet18 -d deconv2 -m rgbd -s 5 --sparsifier uar
#python3 main.py -a resnet18 -d deconv2 -m rgbd -s 10 --sparsifier uar
#python3 main.py -a resnet18 -d deconv2 -m rgbd -s 100 --sparsifier uar
#python3 main.py -a resnet18 -d deconv2 -m rgbd -s 200 --sparsifier uar
#python3 main.py -a resnet18 -d deconv2 -m rgbd -s 1000 --sparsifier uar
#python3 main.py -a resnet18 -d deconv2 -m rgbd -s 10 --sparsifier sim_stereo
#python3 main.py -a resnet18 -d deconv2 -m rgbd -s 100 --sparsifier sim_stereo
#python3 main.py -a resnet18 -d deconv2 -m rgbd -s 1000 --sparsifier sim_stereo
#python3 main.py -a resnet18 -d deconv2 -m rgbd -s 5 --sparsifier  sim_reflector
#python3 main.py -a resnet18 -d deconv2 -m rgbd -s 10 --sparsifier  sim_reflector
#python3 main.py -a resnet18 -d deconv2 -m rgbd -s 20 --sparsifier  sim_reflector
#python3 main.py -a resnet18 -d deconv2 -m rgbd -s 100 --sparsifier  sim_reflector
#python3 main.py -a resnet18 -d deconv2 -m rgbd -s 1000 --sparsifier  sim_reflector
#python3 main.py -a resnet18 -d deconv2 -m rgbw -s 10 --sparsifier sim_wireless
#python3 main.py -a resnet18 -d deconv2 -m rgbw -s 100 --sparsifier sim_wireless
#python3 main.py -a resnet18 -d deconv2 -m rgbw -s 1000 --sparsifier sim_wireless
#
#python3 main.py -a resnet50 -d deconv2 -m rgb -s 0
#python3 main.py -a resnet50 -d deconv3 -m rgbd -s 5 --sparsifier uar
#python3 main.py -a resnet50 -d deconv3 -m rgbd -s 10 --sparsifier uar
#python3 main.py -a resnet50 -d deconv3 -m rgbd -s 100 --sparsifier uar
#python3 main.py -a resnet50 -d deconv3 -m rgbd -s 200 --sparsifier uar
#python3 main.py -a resnet50 -d deconv3 -m rgbd -s 1000 --sparsifier uar
#python3 main.py -a resnet50 -d deconv3 -m rgbd -s 10 --sparsifier sim_stereo
#python3 main.py -a resnet50 -d deconv3 -m rgbd -s 100 --sparsifier sim_stereo
#python3 main.py -a resnet50 -d deconv3 -m rgbd -s 1000 --sparsifier sim_stereo
#python3 main.py -a resnet50 -d deconv3 -m rgbd -s 5 --sparsifier  sim_reflector
#python3 main.py -a resnet50 -d deconv3 -m rgbd -s 10 --sparsifier  sim_reflector
#python3 main.py -a resnet50 -d deconv3 -m rgbd -s 20 --sparsifier  sim_reflector
#python3 main.py -a resnet50 -d deconv3 -m rgbd -s 100 --sparsifier  sim_reflector
#python3 main.py -a resnet50 -d deconv3 -m rgbd -s 1000 --sparsifier  sim_reflector
#python3 main.py -a resnet50 -d deconv3 -m rgbw -s 10 --sparsifier sim_wireless
#python3 main.py -a resnet50 -d deconv3 -m rgbw -s 100 --sparsifier sim_wireless
#python3 main.py -a resnet50 -d deconv3 -m rgbw -s 1000 --sparsifier sim_wireless


## FULL DATASET

## eldar-1 gpu 1
#python3 main.py -a resnet50 -d deconv2 -m rgb -s 0
#python3 main.py -a resnet50 -d deconv3 -m rgbd -s 5 --sparsifier uar
#python3 main.py -a resnet50 -d deconv3 -m rgbd -s 10 --sparsifier uar
#python3 main.py -a resnet50 -d deconv3 -m rgbd -s 100 --sparsifier uar
#python3 main.py -a resnet50 -d deconv3 -m rgbd -s 5 --sparsifier  sim_reflector
#python3 main.py -a resnet50 -d deconv3 -m rgbd -s 10 --sparsifier  sim_reflector
#python3 main.py -a resnet50 -d deconv3 -m rgbd -s 100 --sparsifier  sim_reflector

## eldar-1 gpu 2
#python3 main.py -a resnet50 -d deconv3 -m rgbd -s 5 --sparsifier sim_reflector -g 2
#python3 main.py -a resnet50 -d deconv3 -m rgbd -s 10 --sparsifier sim_reflector -g 2
#python3 main.py -a resnet50 -d deconv3 -m rgbd -s 100 --sparsifier sim_reflector -g 2

## tacc cs398t
python3 main.py -a resnet50 -d upproj -m rgb -s 0
python3 main.py -a resnet50 -d upproj -m rgbd -s 5 --sparsifier uar
python3 main.py -a resnet50 -d upproj -m rgbd -s 10 --sparsifier uar
python3 main.py -a resnet50 -d upproj -m rgbd -s 100 --sparsifier uar

# tacc tracking
#python3 main.py -a resnet50 -d upproj -m rgbd -s 5 --sparsifier  sim_reflector
#python3 main.py -a resnet50 -d upproj -m rgbd -s 10 --sparsifier  sim_reflector
#python3 main.py -a resnet50 -d upproj -m rgbd -s 100 --sparsifier  sim_reflector
