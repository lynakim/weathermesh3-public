#!/bin/bash

# Define the IP and hostname pairs
HOSTS_ENTRIES=(
    "10.0.0.50 stinson.fast"
    "10.0.0.51 halfmoon.fast"
    "10.0.0.52 barceloneta.fast"
    "10.0.0.53 miramar.fast"
    "10.0.0.54 singing.fast"
    "10.0.0.55 bimini.fast"
    "10.0.0.56 baga.fast"
    "10.0.0.57 muir.fast"
    "10.0.0.137 ubuntu-45d.fast"
)

# Define the fstab entries
FSTAB_ENTRIES=(
    "ubuntu-45d.fast:/mcpoolface/huge /huge nfs defaults,fsc,rdma,port=20049,nofail 0 0"
    "halfmoon.fast:/fast /fast nfs defaults,fsc,rdma,port=20049,nofail 0 0"
)

# Function to add an entry to a file if it doesn't already exist
add_entry() {
    local entry="$1"
    local file="$2"
    
    if grep -qF "$entry" "$file"; then
        echo "The entry '$entry' already exists in $file"
    else
        echo "$entry" | sudo tee -a "$file" > /dev/null
        echo "Entry added to $file"
    fi
}

# Add the /etc/hosts entries
for entry in "${HOSTS_ENTRIES[@]}"; do
    add_entry "$entry" "/etc/hosts"
done

# Add the /etc/fstab entries
for entry in "${FSTAB_ENTRIES[@]}"; do
    add_entry "$entry" "/etc/fstab"
done

sudo mkdir /huge /fast
sudo mount -a

./setup_env.sh

pip install -e /fast/optimizers

cd /fast/NEONATTEN
make clean
