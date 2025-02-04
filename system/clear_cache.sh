echo clearing caches 
echo sswa | sudo -S bash -c 'echo 1 > /proc/sys/vm/drop_caches' 
echo page cache cleared