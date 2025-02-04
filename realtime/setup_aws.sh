# Set up the AWS ubuntu instance
# We could probably move this to install.sh but it's not really service specific

# Set up the disk that AWS gives us
sudo mkfs -t ext4 /dev/nvme1n1
sudo mkdir /viz
sudo mount /dev/nvme1n1 /viz

if !grep -q ' /viz ' /etc/fstab ; then
    echo '/dev/nvme1n1 /viz   ext4 defaults,nofail 0 2' >> /etc/fstab
fi

sudo mkfs -t ext4 /dev/nvme2n1
sudo mkdir /fast
sudo mount /dev/nvme2n1 /fast

if !grep -q ' /fast ' /etc/fstab ; then
    echo '/dev/nvme2n1 /fast   ext4 defaults,nofail 0 2' >> /etc/fstab
fi

# Install fish
sudo apt-get update
sudo apt-get install fish
# Optionall: sudo nano /etc/passwd and change the last line to /usr/bin/fish

# Install python 3.10
sudo apt install -y software-properties-common
sudo apt update
sudo apt install -y python3.10 python3-pip
sudo apt install python-is-python3

# Installing aws cli
sudo apt install unzip
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
rm -rf aws/ awscliv2.zip

# Copy ~/.config/gcloud/configurations/config_default and  ~/.config/gcloud/application_default_credentials.json