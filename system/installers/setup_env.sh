#!/bin/bash
set -e

sudo apt install -y software-properties-common
sudo apt update
sudo apt install python-is-python3
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools wheel twine check-wheel-contents
# Setup python packages
pip3 install scipy netCDF4 torch matplotlib tensorboard xarray timm einops gcsfs zarr icosphere psutil GPUtil
pip3 install natten==0.17.1+torch240cu121 -f https://shi-labs.com/natten/wheels/
pip install -e /fast/optimizers

echo "Python env setup done"
