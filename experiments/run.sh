#!/usr/bin/env bash
source ~/Documents/pyenv/tensorflow/bin/activate

for p1 in 0.002 0.001
do
	for p2 in 0.0005 0.0001
	do
		for p3 in 0.0001 0.00005
		do
            echo -e "\033[34mpython resnet_rmsprop.py --p1 $p1 --p2 $p2 --p3 $p3\033[0m"
            python resnet_rmsprop.py --p1 $p1 --p2 $p2 --p3 $p3
		done
	done
done


#echo -e "\033[34mresnet_adam.py --p1 0.001 --p2 0.0001 --p3 0.00005\033[0m"
#python resnet_adam.py --p1 0.001 --p2 0.0001 --p3 0.00005

#echo -e "\033[34mresnet_adam.py --p1 0.01 --p2 0.001 --p3 0.0005\033[0m"
#python resnet_adam.py --p1 0.01 --p2 0.001 --p3 0.0005
#
#
#echo -e "\033[34mresnet_so.py --p1 0.2 --p2 0.05 --p3 0.01 --s 0.01\033[0m"
#python resnet_so.py --p1 0.2 --p2 0.05 --p3 0.01 --s 0.01


#echo -e "\033[34mresnet_so.py --p1 0.5 --p2 0.02 --p3 0.01 --s 0.01\033[0m"
#python resnet_so.py --p1 0.5 --p2 0.02 --p3 0.01 --s 0.01
#
#
#echo -e "\033[34mresnet_so.py --p1 1.0 --p2 0.02 --p3 0.01 --s 0.01\033[0m"
#python resnet_so.py --p1 1.0 --p2 0.02 --p3 0.01 --s 0.01


#echo -e "\033[34mpython resnet_so.py --p1 0.05 --p2 0.02 --p3 0.01 --s 0.01\033[0m"
#python resnet_so.py --p1 0.05 --p2 0.02 --p3 0.01 --s 0.01
#
#
#echo -e "\033[34mpython resnet_so.py --p1 0.1 --p2 0.05 --p3 0.01 --s 0.01\033[0m"
#python resnet_so.py --p1 0.1 --p2 0.05 --p3 0.01 --s 0.01
#
#
#echo -e "\033[34mpython vgg_so.py --p1 0.2 --p2 0.05 --p3 0.005 --r 0.0001 --s 0.01\033[0m"
#python vgg_so.py --p1 0.2 --p2 0.05 --p3 0.005 --r 0.0001 --s 0.01
#
#
#echo -e "\033[34mpython vgg_so.py --p1 2 --p2 0.5 --p3 0.005 --r 0.00001 --s 0.001\033[0m"
#python vgg_so.py --p1 2 --p2 0.5 --p3 0.005 --r 0.00001 --s 0.001


#
#echo -e "\033[34mresnet_so.py --p1 0.05 --p2 0.05 --p3 0.005 --r 0.001 --s 0.1\033[0m"
#python resnet_so.py --p1 0.05 --p2 0.05 --p3 0.005 --r 0.001 --s 0.1
#
#echo -e "\033[34mresnet_so.py --p1 0.05 --p2 0.001 --p3 0.001 --r 0.001 --s 0.1\033[0m"
#python resnet_so.py --p1 0.05 --p2 0.001 --p3 0.001 --r 0.001 --s 0.1
#
#echo -e "\033[34mresnet_so.py --p1 0.05 --p2 0.005 --p3 0.001 --r 0.001 --s 0.1\033[0m"
#python resnet_so.py --p1 0.05 --p2 0.005 --p3 0.001 --r 0.001 --s 0.1


# --------------------------
echo -e "\033[34mTest done!\033[0m"