# Setup
sudo apt update && sudo apt upgrade -y
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y

# Python 3.10
sudo apt install python3.10
sudo apt install python3.10-dev
sudo apt install python3.10-distutils
sudo apt install python3.10-venv

# Pip
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
echo 'export PATH=/home/<USER>/.local/bin:$PATH' >>~/.profile
source ~/.profile
pip install --upgrade pip

# Remap
sudo apt install python-is-python3
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1