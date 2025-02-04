#!/usr/bin/env bash

# This script is badly named, it does not *just* set gpu power limit, it creates another script that sets the power limit and then a system service that calls that created script on boot.
# Before running this, check to see if your host already has the nv-power-limit.sh and you can edit the power limit there if you want.

# Step 1: Create the file /usr/local/sbin/nv-power-limit.sh
cat << 'EOF' | sudo tee /usr/local/sbin/nv-power-limit.sh > /dev/null
#!/usr/bin/env bash

# Set power limits on all NVIDIA GPUs

# Make sure nvidia-smi exists 
command -v nvidia-smi &> /dev/null || { echo >&2 "nvidia-smi not found ... exiting."; exit 1; }

POWER_LIMIT=350   # Should be >150 and <450 usually, check for your GPU
/usr/bin/nvidia-smi --persistence-mode=1
/usr/bin/nvidia-smi --power-limit=${POWER_LIMIT}
exit 0
EOF

# Step 2: Change permissions of the script
sudo chmod 744 /usr/local/sbin/nv-power-limit.sh

# Step 3: Create the directory and service file
sudo mkdir -p /usr/local/etc/systemd

cat << 'EOF' | sudo tee /usr/local/etc/systemd/nv-power-limit.service > /dev/null
[Unit]
Description=NVIDIA GPU Set Power Limit
After=syslog.target systemd-modules-load.service
ConditionPathExists=/usr/bin/nvidia-smi

[Service]
User=root
Environment="PATH=/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
ExecStart=/usr/local/sbin/nv-power-limit.sh

[Install]
WantedBy=multi-user.target
EOF

# Step 4: Change permissions of the service file and create a symlink
sudo chmod 644 /usr/local/etc/systemd/nv-power-limit.service
sudo ln -s /usr/local/etc/systemd/nv-power-limit.service /etc/systemd/system/nv-power-limit.service

# Enable, start, and check the status of the service
sudo systemctl enable nv-power-limit.service
sudo systemctl start nv-power-limit.service
sudo systemctl status nv-power-limit.service

