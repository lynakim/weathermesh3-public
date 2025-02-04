echo "/slow stinson(rw,sync,no_subtree_check)" | sudo tee -a /etc/exports
sudo exportfs -ra

echo "stinson:/fast /fast nfs defaults,fsc 0 0" | sudo tee -a /etc/fstab