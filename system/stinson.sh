echo "/fast martins(rw,sync,no_subtree_check)" | sudo tee -a /etc/exports
sudo exportfs -ra

echo "martins:/slow /slow nfs defaults,fsc 0 0" | sudo tee -a /etc/fstab