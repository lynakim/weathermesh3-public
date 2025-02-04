#!/bin/bash
set -e

# After running this script, copy /fast/consts
# rclone copy /fast/consts nimbus:/fast/consts --progress --transfers 128 --multi-thread-streams 128

sudo ln -s /opt/dlami/nvme /fast
sudo ln -s /fast /huge
sudo mkdir -p /fast/ignored
mkdir -p /huge/deep/runs
git -C /fast clone https://github.com/facebookresearch/optimizers.git

./setup_env.sh


# sudo apt-get install -y python3-grib
exit

# check git history for baton method, automating cmake install, device mapping separate file shares into 1 etc.

