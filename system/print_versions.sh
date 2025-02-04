echo -n "pytorch: "; python3 -c "import torch ; print(torch.__version__)"
echo -n "nccl:    "; python3 -c "import torch;print('.'.join([str(x) for x in torch.cuda.nccl.version()]))"
echo -n "nvidia:  "; nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -n 1
echo -n "kernel:  "; uname -r
echo -n "ubuntu:  "; lsb_release -d | cut -f 2-
echo -n "python:  "; python3 --version 
echo -n "nvcc:    "; nvcc --version | grep release


echo -n "ramspd:  "; echo sswa | sudo -k -S --prompt="" dmidecode --type 17 | grep "Configured Memory Speed" | awk '{print $4}' | sort | uniq
echo -n "nram:    "; echo sswa | sudo -k -S --prompt="" dmidecode --type 17 | grep "Configured Memory Speed" | awk '{print $4}' | wc -l

echo -n "pcie:    "; echo sswa | sudo -k -S --prompt="" lspci -vv | grep -i -E "controller: NVID|controller: mellanox|LnkSta:" | grep NVIDIA -A 1 | grep -P -o "(\d+)GT/s" | tr '\n' ' '; echo
echo -n "width:   "; echo sswa | sudo -k -S --prompt="" lspci -vv | grep -i -E "controller: NVID|controller: mellanox|LnkSta:" | grep NVIDIA -A 1 | grep -P -o "x(\d+)" | tr '\n' ' '; echo
echo -n "ib gbps: "; echo sswa | sudo -k -S --prompt="" ibstat | grep Rate | head -n 1 | awk '{print $2}' ; echo

