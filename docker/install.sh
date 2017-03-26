#!/bin/bash

PROGRAM=$0
USERNAME=$1

set -eu

if [[ $EUID -ne 0 ]]; then
	   echo This script must be run as root
	   exit 1
fi

if [ "$#" -ne 1 ]; then
	    echo USAGE: $PROGRAM USERNAME
	    exit 1
fi

apt-get update
apt-get remove docker docker-engine
apt-get install -y curl wget
curl -fsSL https://get.docker.com/ | sh
usermod -aG docker $USERNAME
wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb
echo NOW REBOOT YOUR SYSTEM
