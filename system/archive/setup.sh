#!/bin/bash
cd "$(dirname "$0")"
cd ../../infra/setup 
source utils.sh


bash iapt.sh
bash ipip.sh
bash ilocalpip.sh
bash ibin.sh
bash idaemon.sh

source ~/.bashrc

pprint "Installing DLNWP-specific stuff"

#sudo timedatectl set-timezone America/Los_Angeles
sudo apt update
#sudo apt install -y nfs-kernel-server nfs-common net-tools python3-pip git
sudo apt install -y iperf3
#sudo apt install -y cachefilesd
sudo apt install -y nginx 
#sudo systemctl enable cachefilesd
#sudo systemctl start cachefilesd

#sudo apt install -y nvidia-cuda-toolkit
#sudo apt install -y stress 

echo "--system-monitor" > ../../infra/tacoma/tacoma_server_args

#pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
pip3 install -r requirements.txt

pushd /fast/optimizers
pip3 install -e .
popd 
